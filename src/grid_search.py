"""
Grid search over clustering hyperparameters
Evaluates: cluster count, size balance, feature homogeneity, silhouette score
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import h3pandas
from h3 import cell_to_latlng
from libpysal.weights import Queen
from sklearn.preprocessing import robust_scale
from sklearn.metrics import silhouette_score
from spopt.region import Skater
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Load and prep data (same as cluster_balanced.py)
print("Loading data...")
manhattan_hexes = pd.read_csv('../data/clean/h3_index/manhattan_hex.csv')['manhattan_hex'].tolist()

df = pd.read_csv('../data/clean/h3_PLUTO.csv', low_memory=False)
df = df[df['h3_10'].isin(manhattan_hexes)]

pluto_agg = df.groupby('h3_10').agg({
    'BldgArea': 'sum', 'YearBuilt': 'mean', 'AssessTot': 'sum',
    'ResArea': 'sum', 'ComArea': 'sum', 'RetailArea': 'sum', 'UnitsRes': 'sum',
}).reset_index()
pluto_agg = pluto_agg[pluto_agg['YearBuilt'] > 1500]

poi_df = pd.read_csv('../data/clean/poi_h3.csv')
poi_counts = poi_df[poi_df['h3_10'].isin(manhattan_hexes)].groupby('h3_10').size().reset_index(name='poi_count')
df_gb = pluto_agg.merge(poi_counts, on='h3_10', how='left').fillna({'poi_count': 0})

external_df = pd.read_csv('../data/clean/external_features_h3.csv')
df_gb = df_gb.merge(external_df, on='h3_10', how='left')

for col in ['subway_dist_m', 'crime_count', 'complaint_311_count']:
    df_gb[col] = df_gb[col].fillna(df_gb[col].median())

airbnb_cols = ['airbnb_count', 'airbnb_reviews_total', 'airbnb_entire_home_apt',
               'airbnb_private_room', 'airbnb_shared_room', 'airbnb_hotel_room']
for col in airbnb_cols:
    if col in df_gb.columns:
        df_gb[col] = df_gb[col].fillna(0)

school_cols = ['school_count', 'school_enrollment_total']
for col in school_cols:
    if col in df_gb.columns:
        df_gb[col] = df_gb[col].fillna(0)

# Log transform
skewed_features = ['BldgArea', 'AssessTot', 'ResArea', 'ComArea', 'RetailArea', 'UnitsRes',
                   'crime_count', 'complaint_311_count',
                   'airbnb_count', 'airbnb_reviews_total', 'airbnb_entire_home_apt', 'airbnb_private_room',
                   'school_enrollment_total']
for f in skewed_features:
    df_gb[f'{f}_log'] = np.log1p(df_gb[f])

cluster_variables_log = [
    'BldgArea_log', 'YearBuilt', 'AssessTot_log', 'ResArea_log',
    'ComArea_log', 'RetailArea_log', 'UnitsRes_log', 'poi_count',
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

print(f"Clustering {len(gdf_clean)} hexagons\n")

# Prepare feature matrix for metrics
X = gdf_clean[cluster_variables_log].values
X_scaled = robust_scale(X)


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
            gdf_subset.reset_index(drop=True), w_sub, features,
            n_clusters=n_clusters, floor=min_size, trace=False, islands='ignore'
        )
        skater.solve()
        return skater.labels_
    except:
        return np.zeros(len(gdf_subset))


def run_clustering(n_clusters, min_size, max_size):
    """Run full clustering pipeline and return labels"""
    gdf_temp = gdf_clean.copy()

    try:
        skater = Skater(
            gdf_temp, w_clean, cluster_variables_log,
            n_clusters=n_clusters, floor=min_size, trace=False, islands='ignore'
        )
        skater.solve()
        gdf_temp['cluster'] = skater.labels_
    except Exception as e:
        return None, str(e)

    # Phase 2: Split large clusters
    next_cluster_id = gdf_temp['cluster'].max() + 1
    for iteration in range(15):
        sizes = gdf_temp['cluster'].value_counts()
        large_clusters = sizes[sizes > max_size].index.tolist()

        if not large_clusters:
            break

        for cluster_id in large_clusters:
            mask = gdf_temp['cluster'] == cluster_id
            subset = gdf_temp[mask].copy()
            n_sub = max(2, int(np.ceil(len(subset) / max_size)))

            sub_labels = run_skater_on_subset(subset, n_sub, min_size, cluster_variables_log)

            for sub_id in np.unique(sub_labels):
                if sub_id == 0:
                    continue
                sub_mask = sub_labels == sub_id
                gdf_temp.loc[mask, 'cluster'] = np.where(
                    sub_mask, next_cluster_id, gdf_temp.loc[mask, 'cluster'].values
                )
                next_cluster_id += 1

    return gdf_temp['cluster'].values, None


def compute_metrics(labels):
    """Compute all evaluation metrics"""
    if labels is None:
        return None

    # Filter out any -1 labels
    valid_mask = labels >= 0
    labels_valid = labels[valid_mask]
    X_valid = X_scaled[valid_mask]

    n_clusters = len(np.unique(labels_valid))
    sizes = pd.Series(labels_valid).value_counts()

    # 1. Cluster count
    cluster_count = n_clusters

    # 2. Size balance (coefficient of variation - lower is better)
    size_cv = sizes.std() / sizes.mean() if sizes.mean() > 0 else float('inf')

    # 3. Within-cluster variance (lower is better)
    within_var = 0
    for c in np.unique(labels_valid):
        cluster_mask = labels_valid == c
        if cluster_mask.sum() > 1:
            cluster_data = X_valid[cluster_mask]
            within_var += np.var(cluster_data, axis=0).sum()
    within_var /= n_clusters

    # 4. Silhouette score (higher is better, -1 to 1)
    if n_clusters > 1 and n_clusters < len(labels_valid):
        try:
            sil_score = silhouette_score(X_valid, labels_valid)
        except:
            sil_score = float('nan')
    else:
        sil_score = float('nan')

    return {
        'n_clusters': cluster_count,
        'size_min': sizes.min(),
        'size_max': sizes.max(),
        'size_mean': sizes.mean(),
        'size_cv': size_cv,
        'within_var': within_var,
        'silhouette': sil_score,
    }


# Grid search parameters
param_grid = {
    'n_clusters': [6, 8, 10, 12, 15],
    'min_size': [20, 30, 40, 50],
    'max_size': [150, 200, 250, 300],
}

results = []

print("=" * 80)
print("GRID SEARCH")
print("=" * 80)

total = len(param_grid['n_clusters']) * len(param_grid['min_size']) * len(param_grid['max_size'])
i = 0

for n_clusters in param_grid['n_clusters']:
    for min_size in param_grid['min_size']:
        for max_size in param_grid['max_size']:
            i += 1
            print(f"\n[{i}/{total}] n_clusters={n_clusters}, min_size={min_size}, max_size={max_size}")

            labels, error = run_clustering(n_clusters, min_size, max_size)

            if error:
                print(f"  ERROR: {error}")
                continue

            metrics = compute_metrics(labels)
            if metrics:
                metrics['n_clusters_init'] = n_clusters
                metrics['min_size'] = min_size
                metrics['max_size'] = max_size
                results.append(metrics)

                print(f"  -> {metrics['n_clusters']} clusters, "
                      f"sizes {metrics['size_min']}-{metrics['size_max']}, "
                      f"CV={metrics['size_cv']:.3f}, "
                      f"silhouette={metrics['silhouette']:.3f}")

# Summary
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df.to_csv('../out/grid_search_results.csv', index=False)
print(f"\nSaved {len(results_df)} results to ../out/grid_search_results.csv")

# Best by different criteria
print("\n--- Best by Silhouette Score (feature separation) ---")
best_sil = results_df.loc[results_df['silhouette'].idxmax()]
print(best_sil.to_string())

print("\n--- Best by Size Balance (lowest CV) ---")
best_cv = results_df.loc[results_df['size_cv'].idxmin()]
print(best_cv.to_string())

print("\n--- Best by Within-Cluster Variance (homogeneity) ---")
best_var = results_df.loc[results_df['within_var'].idxmin()]
print(best_var.to_string())

print("\n--- Closest to 25 clusters with good balance ---")
target_25 = results_df[results_df['n_clusters'].between(23, 27)].sort_values('size_cv')
if len(target_25) > 0:
    print(target_25.head(3).to_string())
else:
    print("No results with 23-27 clusters")

print("\n--- Top 10 by combined score (normalized) ---")
# Normalize metrics (0-1, with direction so higher is better)
results_df['sil_norm'] = (results_df['silhouette'] - results_df['silhouette'].min()) / (results_df['silhouette'].max() - results_df['silhouette'].min())
results_df['cv_norm'] = 1 - (results_df['size_cv'] - results_df['size_cv'].min()) / (results_df['size_cv'].max() - results_df['size_cv'].min())
results_df['var_norm'] = 1 - (results_df['within_var'] - results_df['within_var'].min()) / (results_df['within_var'].max() - results_df['within_var'].min())
results_df['combined'] = results_df['sil_norm'] + results_df['cv_norm'] + results_df['var_norm']

print(results_df.nlargest(10, 'combined')[['n_clusters_init', 'min_size', 'max_size', 'n_clusters',
                                            'size_min', 'size_max', 'size_cv', 'silhouette', 'combined']].to_string())
