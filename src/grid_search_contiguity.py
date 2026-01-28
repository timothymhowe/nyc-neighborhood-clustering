"""
Grid search optimizing for cluster contiguity
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import h3pandas
from h3 import cell_to_latlng
from libpysal.weights import Queen
from spopt.region import Skater
import networkx as nx
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data (same as cluster_balanced.py)
print("Loading data...")
manhattan_hexes = pd.read_csv('../data/clean/h3_index/manhattan_hex.csv')['manhattan_hex'].tolist()

df = pd.read_csv('../data/clean/h3_PLUTO.csv', low_memory=False)
df = df[df['h3_10'].isin(manhattan_hexes)]

df['is_residential'] = df['LandUse'].isin([1.0, 2.0, 3.0])
df['is_commercial'] = df['LandUse'].isin([4.0, 5.0, 6.0])
df['AssessTot_res'] = df['AssessTot'].where(df['is_residential'], 0)
df['AssessTot_com'] = df['AssessTot'].where(df['is_commercial'], 0)

pluto_agg = df.groupby('h3_10').agg({
    'BldgArea': 'sum', 'YearBuilt': 'mean', 'AssessTot': 'sum',
    'AssessTot_res': 'sum', 'AssessTot_com': 'sum',
    'ResArea': 'sum', 'ComArea': 'sum', 'RetailArea': 'sum', 'UnitsRes': 'sum',
}).reset_index()

valid_year_median = pluto_agg[pluto_agg['YearBuilt'] > 1500]['YearBuilt'].median()
pluto_agg.loc[pluto_agg['YearBuilt'] <= 1500, 'YearBuilt'] = valid_year_median

pluto_agg = pluto_agg[
    (pluto_agg['BldgArea'] > 0) |
    (pluto_agg['AssessTot'] > 0) |
    (pluto_agg['ResArea'] > 0) |
    (pluto_agg['ComArea'] > 0)
]

# $/sqft features
pluto_agg['res_dollar_sqft'] = np.where(pluto_agg['ResArea'] > 0, pluto_agg['AssessTot_res'] / pluto_agg['ResArea'], 0)
pluto_agg['com_dollar_sqft'] = np.where(pluto_agg['ComArea'] > 0, pluto_agg['AssessTot_com'] / pluto_agg['ComArea'], 0)
pluto_agg['units_per_1k_sqft'] = np.where(pluto_agg['ResArea'] > 0, pluto_agg['UnitsRes'] / (pluto_agg['ResArea'] / 1000), 0)

poi_df = pd.read_csv('../data/clean/poi_h3.csv')
poi_counts = poi_df[poi_df['h3_10'].isin(manhattan_hexes)].groupby('h3_10').size().reset_index(name='poi_count')
df_gb = pluto_agg.merge(poi_counts, on='h3_10', how='left').fillna({'poi_count': 0})

external_df = pd.read_csv('../data/clean/external_features_h3.csv')
df_gb = df_gb.merge(external_df, on='h3_10', how='left')

for col in ['subway_dist_m', 'crime_count', 'complaint_311_count']:
    df_gb[col] = df_gb[col].fillna(df_gb[col].median())

airbnb_cols = ['airbnb_count', 'airbnb_reviews_total', 'airbnb_entire_home_apt', 'airbnb_private_room', 'airbnb_shared_room', 'airbnb_hotel_room']
for col in airbnb_cols:
    if col in df_gb.columns:
        df_gb[col] = df_gb[col].fillna(0)

school_cols = ['school_count', 'school_enrollment_total']
for col in school_cols:
    if col in df_gb.columns:
        df_gb[col] = df_gb[col].fillna(0)

df_gb['lat'], df_gb['lon'] = zip(*df_gb['h3_10'].apply(cell_to_latlng))

# Log transform
skewed_features = ['BldgArea', 'AssessTot', 'ResArea', 'ComArea', 'RetailArea', 'UnitsRes',
                   'crime_count', 'complaint_311_count',
                   'airbnb_count', 'airbnb_reviews_total', 'airbnb_entire_home_apt', 'airbnb_private_room',
                   'school_enrollment_total', 'res_dollar_sqft', 'com_dollar_sqft']
for f in skewed_features:
    df_gb[f'{f}_log'] = np.log1p(df_gb[f])

cluster_variables_log = [
    'BldgArea_log', 'YearBuilt', 'AssessTot_log', 'ResArea_log',
    'ComArea_log', 'RetailArea_log', 'UnitsRes_log', 'poi_count',
    'res_dollar_sqft_log', 'com_dollar_sqft_log',
    'subway_dist_m', 'crime_count_log', 'complaint_311_count_log',
    'airbnb_count_log', 'airbnb_reviews_total_log', 'airbnb_entire_home_apt_log', 'airbnb_private_room_log',
    'school_count', 'school_enrollment_total_log',
]

# Build spatial structure
print("Building spatial structure...")
gdf = df_gb.set_index('h3_10').h3.h3_to_geo_boundary().reset_index()
gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')

w = Queen.from_dataframe(gdf)
G = w.to_networkx()
largest = max(nx.connected_components(G), key=len)
gdf_clean = gdf.iloc[list(largest)].reset_index(drop=True)
w_clean = Queen.from_dataframe(gdf_clean)

print(f"Working with {len(gdf_clean)} hexagons")


def count_cluster_fragments(gdf_with_clusters, w):
    """Count total number of disconnected components across all clusters"""
    G = w.to_networkx()
    total_fragments = 0

    for cluster_id in gdf_with_clusters['cluster'].unique():
        if cluster_id < 0:
            continue
        cluster_nodes = gdf_with_clusters[gdf_with_clusters['cluster'] == cluster_id].index.tolist()
        subgraph = G.subgraph(cluster_nodes)
        n_components = nx.number_connected_components(subgraph)
        total_fragments += n_components

    return total_fragments


def run_skater_on_subset(gdf_subset, n_clusters, min_size, features):
    """Run SKATER on a subset of hexagons"""
    if len(gdf_subset) < min_size * 2:
        return np.zeros(len(gdf_subset))

    w_sub = Queen.from_dataframe(gdf_subset.reset_index(drop=True))
    G_sub = w_sub.to_networkx()
    if not nx.is_connected(G_sub):
        return np.zeros(len(gdf_subset))

    try:
        skater = Skater(
            gdf_subset.reset_index(drop=True),
            w_sub,
            features,
            n_clusters=n_clusters,
            floor=min_size,
            trace=False,
            islands='ignore'
        )
        skater.solve()
        return skater.labels_
    except:
        return np.zeros(len(gdf_subset))


def run_full_clustering(gdf_clean, w_clean, n_clusters, min_size, max_size, features):
    """Run SKATER with recursive splitting"""
    gdf_work = gdf_clean.copy()

    try:
        skater = Skater(
            gdf_work, w_clean, features,
            n_clusters=n_clusters, floor=min_size, trace=False, islands='ignore'
        )
        skater.solve()
        gdf_work['cluster'] = skater.labels_
    except Exception as e:
        return None, str(e)

    # Split large clusters
    next_cluster_id = gdf_work['cluster'].max() + 1
    for _ in range(15):
        sizes = gdf_work['cluster'].value_counts()
        large_clusters = sizes[sizes > max_size].index.tolist()
        if not large_clusters:
            break

        for cluster_id in large_clusters:
            mask = gdf_work['cluster'] == cluster_id
            subset = gdf_work[mask].copy()
            n_sub = max(2, int(np.ceil(len(subset) / max_size)))
            sub_labels = run_skater_on_subset(subset, n_sub, min_size, features)

            for sub_id in np.unique(sub_labels):
                if sub_id == 0:
                    continue
                sub_mask = sub_labels == sub_id
                gdf_work.loc[mask, 'cluster'] = np.where(
                    sub_mask, next_cluster_id, gdf_work.loc[mask, 'cluster'].values
                )
                next_cluster_id += 1

    return gdf_work, None


# Grid search
print("\n" + "="*60)
print("GRID SEARCH: Optimizing for contiguity")
print("="*60)

results = []

# Parameters to try
n_clusters_options = [4, 5, 6, 7, 8]
min_size_options = [30, 40, 50]
max_size_options = [100, 120, 150, 180]

total_combos = len(n_clusters_options) * len(min_size_options) * len(max_size_options)
combo_num = 0

for n_clusters in n_clusters_options:
    for min_size in min_size_options:
        for max_size in max_size_options:
            combo_num += 1
            print(f"\n[{combo_num}/{total_combos}] n={n_clusters}, min={min_size}, max={max_size}")

            gdf_result, error = run_full_clustering(
                gdf_clean.copy(), w_clean, n_clusters, min_size, max_size, cluster_variables_log
            )

            if error:
                print(f"  ERROR: {error}")
                continue

            # Metrics
            final_clusters = len(gdf_result['cluster'].unique())
            sizes = gdf_result['cluster'].value_counts()
            size_cv = sizes.std() / sizes.mean()  # Coefficient of variation

            # Contiguity: count fragments
            w_result = Queen.from_dataframe(gdf_result.reset_index(drop=True))
            fragments = count_cluster_fragments(gdf_result.reset_index(drop=True), w_result)

            # Silhouette score
            try:
                X = gdf_result[cluster_variables_log].values
                sil = silhouette_score(X, gdf_result['cluster'])
            except:
                sil = -1

            results.append({
                'n_clusters': n_clusters,
                'min_size': min_size,
                'max_size': max_size,
                'final_clusters': final_clusters,
                'fragments': fragments,
                'extra_fragments': fragments - final_clusters,  # 0 = perfect contiguity
                'size_cv': size_cv,
                'silhouette': sil,
                'min_cluster_size': sizes.min(),
                'max_cluster_size': sizes.max(),
            })

            print(f"  → {final_clusters} clusters, {fragments} fragments ({fragments - final_clusters} extra), sil={sil:.3f}")

# Results summary
print("\n" + "="*60)
print("RESULTS")
print("="*60)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('extra_fragments')

print("\nTop 10 by contiguity (fewest extra fragments):")
print(df_results.head(10).to_string(index=False))

print("\n\nTop 10 by silhouette score:")
print(df_results.sort_values('silhouette', ascending=False).head(10).to_string(index=False))

# Best compromise
df_results['score'] = -df_results['extra_fragments'] * 10 + df_results['silhouette'] * 100 - df_results['size_cv'] * 5
print("\n\nTop 10 by combined score (contiguity + silhouette - size_cv):")
print(df_results.sort_values('score', ascending=False).head(10).to_string(index=False))

df_results.to_csv('../out/grid_search_contiguity.csv', index=False)
print("\nSaved results to grid_search_contiguity.csv")
