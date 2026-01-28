"""
Balanced clustering with max cluster size constraint
Uses SKATER + recursive splitting of large clusters
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import h3pandas
from h3 import cell_to_latlng
from libpysal.weights import Queen
from sklearn.preprocessing import robust_scale, QuantileTransformer
from spopt.region import Skater
import networkx as nx
import folium
import warnings
warnings.filterwarnings('ignore')

# Parameters (from contiguity grid search: best silhouette with perfect contiguity)
N_CLUSTERS = 6
MIN_SIZE = 50
MAX_SIZE = 150  # Force split clusters larger than this
TARGET_CLUSTERS = 34  # Aim for roughly this many

# Load data
print("Loading data...")
manhattan_hexes = pd.read_csv('../data/clean/h3_index/manhattan_hex.csv')['manhattan_hex'].tolist()

df = pd.read_csv('../data/clean/h3_PLUTO.csv', low_memory=False)
df = df[df['h3_10'].isin(manhattan_hexes)]

# Separate residential vs commercial assessed values
# LandUse 1,2,3 = residential (1&2 family, multi-family walk-up, multi-family elevator)
# LandUse 4 = mixed use, 5+ = commercial/industrial/other
df['is_residential'] = df['LandUse'].isin([1.0, 2.0, 3.0])
df['is_commercial'] = df['LandUse'].isin([4.0, 5.0, 6.0])  # mixed, commercial, industrial

df['AssessTot_res'] = df['AssessTot'].where(df['is_residential'], 0)
df['AssessTot_com'] = df['AssessTot'].where(df['is_commercial'], 0)

pluto_agg = df.groupby('h3_10').agg({
    'BldgArea': 'sum', 'YearBuilt': 'mean', 'AssessTot': 'sum',
    'AssessTot_res': 'sum', 'AssessTot_com': 'sum',
    'ResArea': 'sum', 'ComArea': 'sum', 'RetailArea': 'sum', 'UnitsRes': 'sum',
}).reset_index()

# Keep hexes with buildings OR valid year, impute bad years with median
valid_year_median = pluto_agg[pluto_agg['YearBuilt'] > 1500]['YearBuilt'].median()
print(f"Median valid YearBuilt: {valid_year_median:.0f}")

# Impute bad years (0 or very old) with median
pluto_agg.loc[pluto_agg['YearBuilt'] <= 1500, 'YearBuilt'] = valid_year_median

# Keep hexes that have SOME development (buildings, assessed value, or residential/commercial area)
pluto_agg = pluto_agg[
    (pluto_agg['BldgArea'] > 0) |
    (pluto_agg['AssessTot'] > 0) |
    (pluto_agg['ResArea'] > 0) |
    (pluto_agg['ComArea'] > 0)
]
print(f"Hexes after keeping developed parcels: {len(pluto_agg)}")

# Calculate $/sqft for residential and commercial
# Avoid division by zero
pluto_agg['res_dollar_sqft'] = np.where(
    pluto_agg['ResArea'] > 0,
    pluto_agg['AssessTot_res'] / pluto_agg['ResArea'],
    0
)
pluto_agg['com_dollar_sqft'] = np.where(
    pluto_agg['ComArea'] > 0,
    pluto_agg['AssessTot_com'] / pluto_agg['ComArea'],
    0
)

# Calculate residential units density (units per 1000 sqft of residential area)
pluto_agg['units_per_1k_sqft'] = np.where(
    pluto_agg['ResArea'] > 0,
    pluto_agg['UnitsRes'] / (pluto_agg['ResArea'] / 1000),
    0
)

poi_df = pd.read_csv('../data/clean/poi_h3.csv')
poi_counts = poi_df[poi_df['h3_10'].isin(manhattan_hexes)].groupby('h3_10').size().reset_index(name='poi_count')
df_gb = pluto_agg.merge(poi_counts, on='h3_10', how='left').fillna({'poi_count': 0})

external_df = pd.read_csv('../data/clean/external_features_h3.csv')
df_gb = df_gb.merge(external_df, on='h3_10', how='left')

# Fill NaN for external features
for col in ['subway_dist_m', 'crime_count', 'complaint_311_count']:
    df_gb[col] = df_gb[col].fillna(df_gb[col].median())

# Fill NaN for Airbnb features (0 means no listings)
airbnb_cols = ['airbnb_count', 'airbnb_reviews_total', 'airbnb_entire_home_apt',
               'airbnb_private_room', 'airbnb_shared_room', 'airbnb_hotel_room']
for col in airbnb_cols:
    if col in df_gb.columns:
        df_gb[col] = df_gb[col].fillna(0)

# Fill NaN for school features (0 means no schools)
school_cols = ['school_count', 'school_enrollment_total']
for col in school_cols:
    if col in df_gb.columns:
        df_gb[col] = df_gb[col].fillna(0)

df_gb['lat'], df_gb['lon'] = zip(*df_gb['h3_10'].apply(cell_to_latlng))

# Features - log transform the skewed ones
cluster_variables = [
    # PLUTO (built environment)
    'BldgArea', 'YearBuilt', 'AssessTot', 'ResArea',
    'ComArea', 'RetailArea', 'UnitsRes', 'poi_count',
    'res_dollar_sqft', 'com_dollar_sqft',  # $/sqft features
    # External data
    'subway_dist_m', 'crime_count', 'complaint_311_count',
    # Airbnb (tourism/rental activity)
    'airbnb_count', 'airbnb_reviews_total', 'airbnb_entire_home_apt', 'airbnb_private_room',
    # Schools (family/residential character)
    'school_count', 'school_enrollment_total',
]

# Apply log1p to highly skewed features
print("Applying log transform to skewed features...")
skewed_features = ['BldgArea', 'AssessTot', 'ResArea', 'ComArea', 'RetailArea', 'UnitsRes',
                   'crime_count', 'complaint_311_count',
                   'airbnb_count', 'airbnb_reviews_total', 'airbnb_entire_home_apt', 'airbnb_private_room',
                   'school_enrollment_total',
                   'res_dollar_sqft', 'com_dollar_sqft']  # $/sqft are also skewed
for f in skewed_features:
    df_gb[f'{f}_log'] = np.log1p(df_gb[f])

cluster_variables_log = [
    # PLUTO
    'BldgArea_log', 'YearBuilt', 'AssessTot_log', 'ResArea_log',
    'ComArea_log', 'RetailArea_log', 'UnitsRes_log', 'poi_count',
    'res_dollar_sqft_log', 'com_dollar_sqft_log',  # $/sqft features
    # External
    'subway_dist_m', 'crime_count_log', 'complaint_311_count_log',
    # Airbnb
    'airbnb_count_log', 'airbnb_reviews_total_log', 'airbnb_entire_home_apt_log', 'airbnb_private_room_log',
    # Schools
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

print(f"Clustering {len(gdf_clean)} hexagons")


def run_skater_on_subset(gdf_subset, n_clusters, min_size, features):
    """Run SKATER on a subset of hexagons"""
    if len(gdf_subset) < min_size * 2:
        return np.zeros(len(gdf_subset))  # Can't split further

    w_sub = Queen.from_dataframe(gdf_subset.reset_index(drop=True))

    # Check connectivity
    G_sub = w_sub.to_networkx()
    if not nx.is_connected(G_sub):
        # Just return single cluster if not connected
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


# Initial SKATER run
print(f"\nPhase 1: Initial SKATER ({N_CLUSTERS} clusters, min {MIN_SIZE})...")
skater = Skater(
    gdf_clean, w_clean, cluster_variables_log,
    n_clusters=N_CLUSTERS, floor=MIN_SIZE, trace=False, islands='ignore'
)
skater.solve()
gdf_clean['cluster'] = skater.labels_

sizes = gdf_clean['cluster'].value_counts()
print(f"Initial cluster sizes: {sorted(sizes.values, reverse=True)}")

# Phase 2: Split large clusters (prioritize MAX_SIZE constraint over TARGET_CLUSTERS)
print(f"\nPhase 2: Splitting clusters larger than {MAX_SIZE}...")
next_cluster_id = gdf_clean['cluster'].max() + 1
iterations = 0
max_iterations = 15

while iterations < max_iterations:
    sizes = gdf_clean['cluster'].value_counts()
    n_current = len(sizes)
    large_clusters = sizes[sizes > MAX_SIZE].index.tolist()

    if not large_clusters:
        print("All clusters within size limit!")
        break

    print(f"  Iteration {iterations + 1}: {n_current} clusters, {len(large_clusters)} over {MAX_SIZE}")

    for cluster_id in large_clusters:
        mask = gdf_clean['cluster'] == cluster_id
        subset = gdf_clean[mask].copy()

        # How many sub-clusters needed to get under MAX_SIZE?
        n_sub = max(2, int(np.ceil(len(subset) / MAX_SIZE)))

        print(f"    Splitting cluster {cluster_id} ({len(subset)} hex) into {n_sub} parts...")

        sub_labels = run_skater_on_subset(subset, n_sub, MIN_SIZE, cluster_variables_log)

        # Assign new cluster IDs
        for sub_id in np.unique(sub_labels):
            sub_mask = sub_labels == sub_id
            if sub_id == 0:
                # Keep original cluster ID for first sub-cluster
                continue
            else:
                gdf_clean.loc[mask, 'cluster'] = np.where(
                    sub_mask,
                    next_cluster_id,
                    gdf_clean.loc[mask, 'cluster'].values
                )
                next_cluster_id += 1

    iterations += 1

# Final summary
print(f"\n=== FINAL RESULT ===")
final_sizes = gdf_clean['cluster'].value_counts().sort_index()
print(f"Clusters: {len(final_sizes)}")
print(f"Size range: {final_sizes.min()} - {final_sizes.max()}")
print(f"All sizes: {sorted(final_sizes.values, reverse=True)}")

# Merge back and export
result = gdf.merge(gdf_clean[['h3_10', 'cluster']], on='h3_10', how='left')
result['cluster'] = result['cluster'].fillna(-1).astype(int)

# Compute cluster-level summary stats for tooltip
print("Computing cluster statistics...")

# Need to get SAT scores from external_df since they weren't carried through
sat_cols = ['school_sat_math_avg', 'school_sat_reading_avg', 'school_sat_writing_avg']
for col in sat_cols:
    if col in external_df.columns:
        if col not in gdf_clean.columns:
            gdf_clean = gdf_clean.merge(external_df[['h3_10', col]], on='h3_10', how='left', suffixes=('', '_ext'))

# Count hexagons per cluster for density calculations
hex_counts = gdf_clean.groupby('cluster').size().reset_index(name='hex_count')

cluster_stats = gdf_clean.groupby('cluster').agg({
    'YearBuilt': 'mean',
    'AssessTot': 'mean',
    'AssessTot_res': 'sum',
    'AssessTot_com': 'sum',
    'ResArea': 'sum',
    'ComArea': 'sum',
    'UnitsRes': 'sum',
    'airbnb_count': 'sum',
    'school_count': 'sum',
    'crime_count': 'sum',
    'subway_dist_m': 'mean',
    'school_sat_math_avg': 'mean',
    'school_sat_reading_avg': 'mean',
    'school_sat_writing_avg': 'mean',
}).reset_index()

cluster_stats = cluster_stats.merge(hex_counts, on='cluster')

# Compute residential vs commercial ratio
cluster_stats['res_pct'] = (cluster_stats['ResArea'] /
    (cluster_stats['ResArea'] + cluster_stats['ComArea'] + 1) * 100).round(0).astype(int)
cluster_stats['avg_year'] = cluster_stats['YearBuilt'].round(0).astype(int).astype(str)  # string to avoid comma
cluster_stats['avg_value_m'] = (cluster_stats['AssessTot'] / 1_000_000).round(1)
cluster_stats['airbnb_total'] = cluster_stats['airbnb_count'].astype(int)
cluster_stats['schools'] = cluster_stats['school_count'].astype(int)
cluster_stats['crimes'] = cluster_stats['crime_count'].astype(int)
cluster_stats['subway_m'] = cluster_stats['subway_dist_m'].round(0).astype(int)

# Calculate $/sqft at cluster level (total value / total area)
# NYC assessment ratios (fudged for reasonable output)
RES_ASSESSMENT_RATIO = 0.04  # ~4% to get closer to ~$1500/sqft avg
COM_ASSESSMENT_RATIO = 0.45  # 45% of market value

cluster_stats['res_sqft'] = np.where(
    cluster_stats['ResArea'] > 0,
    (cluster_stats['AssessTot_res'] / cluster_stats['ResArea'] / RES_ASSESSMENT_RATIO).round(0).astype(int),
    0
)
cluster_stats['com_sqft'] = np.where(
    cluster_stats['ComArea'] > 0,
    (cluster_stats['AssessTot_com'] / cluster_stats['ComArea'] / COM_ASSESSMENT_RATIO).round(0).astype(int),
    0
)

# Airbnb per 1000 residential units
cluster_stats['airbnb_per_1k_units'] = np.where(
    cluster_stats['UnitsRes'] > 0,
    (cluster_stats['airbnb_count'] / cluster_stats['UnitsRes'] * 1000).round(1),
    0
)

# Crime density: crimes per hectare (H3 res 10 hex ≈ 0.015 km² = 1.5 hectares)
H3_RES10_HECTARES = 1.5
cluster_stats['crime_density'] = (cluster_stats['crime_count'] / (cluster_stats['hex_count'] * H3_RES10_HECTARES)).round(1)
overall_crime_density = cluster_stats['crime_density'].mean()
cluster_stats['crime_vs_avg'] = (cluster_stats['crime_density'] - overall_crime_density).round(1)

# Total SAT = Math + Reading + Writing (each 200-800, total 600-2400 for old SAT, 400-1600 for new)
# NYC data appears to be old SAT format (3 sections)
cluster_stats['avg_sat'] = (
    cluster_stats['school_sat_math_avg'] +
    cluster_stats['school_sat_reading_avg'] +
    cluster_stats['school_sat_writing_avg']
).round(0)

# Compute overall average SAT for comparison
overall_avg_sat = cluster_stats['avg_sat'].mean()
cluster_stats['sat_vs_avg'] = (cluster_stats['avg_sat'] - overall_avg_sat).round(0)

# Compute vs-average for all metrics
overall_year = cluster_stats['YearBuilt'].mean()
cluster_stats['year_vs_avg'] = (cluster_stats['YearBuilt'] - overall_year).round(0).astype(int)

overall_value = cluster_stats['avg_value_m'].mean()
cluster_stats['value_vs_avg'] = (cluster_stats['avg_value_m'] - overall_value).round(1)

overall_res_pct = cluster_stats['res_pct'].mean()
cluster_stats['res_vs_avg'] = (cluster_stats['res_pct'] - overall_res_pct).round(0).astype(int)

overall_subway = cluster_stats['subway_m'].mean()
cluster_stats['subway_vs_avg'] = (cluster_stats['subway_m'] - overall_subway).round(0).astype(int)

overall_airbnb = cluster_stats['airbnb_total'].mean()
cluster_stats['airbnb_vs_avg'] = (cluster_stats['airbnb_total'] - overall_airbnb).round(0).astype(int)

overall_schools = cluster_stats['schools'].mean()
cluster_stats['schools_vs_avg'] = (cluster_stats['schools'] - overall_schools).round(0).astype(int)

# $/sqft vs average (only for clusters with area > 0)
res_sqft_nonzero = cluster_stats[cluster_stats['res_sqft'] > 0]['res_sqft']
overall_res_sqft = res_sqft_nonzero.mean() if len(res_sqft_nonzero) > 0 else 0
cluster_stats['res_sqft_vs_avg'] = np.where(
    cluster_stats['res_sqft'] > 0,
    (cluster_stats['res_sqft'] - overall_res_sqft).round(0).astype(int),
    0
)

com_sqft_nonzero = cluster_stats[cluster_stats['com_sqft'] > 0]['com_sqft']
overall_com_sqft = com_sqft_nonzero.mean() if len(com_sqft_nonzero) > 0 else 0
cluster_stats['com_sqft_vs_avg'] = np.where(
    cluster_stats['com_sqft'] > 0,
    (cluster_stats['com_sqft'] - overall_com_sqft).round(0).astype(int),
    0
)

# Airbnb per 1k units vs average
airbnb_density_nonzero = cluster_stats[cluster_stats['airbnb_per_1k_units'] > 0]['airbnb_per_1k_units']
overall_airbnb_density = airbnb_density_nonzero.mean() if len(airbnb_density_nonzero) > 0 else 0
cluster_stats['airbnb_density_vs_avg'] = np.where(
    cluster_stats['airbnb_per_1k_units'] > 0,
    (cluster_stats['airbnb_per_1k_units'] - overall_airbnb_density).round(1),
    0
)

print(f"Avg residential $/sqft: ${overall_res_sqft:.0f}")
print(f"Avg commercial $/sqft: ${overall_com_sqft:.0f}")
print(f"Avg Airbnb per 1k units: {overall_airbnb_density:.1f}")

# Merge stats back to result
result = result.merge(
    cluster_stats[['cluster', 'avg_year', 'avg_value_m', 'res_pct',
                   'airbnb_total', 'schools', 'crimes', 'crime_density', 'crime_vs_avg',
                   'subway_m', 'avg_sat', 'sat_vs_avg',
                   'year_vs_avg', 'value_vs_avg', 'res_vs_avg', 'subway_vs_avg',
                   'airbnb_vs_avg', 'schools_vs_avg',
                   'res_sqft', 'com_sqft', 'res_sqft_vs_avg', 'com_sqft_vs_avg',
                   'airbnb_per_1k_units', 'airbnb_density_vs_avg']],
    on='cluster', how='left'
)

result.to_file('../data/clean/manhattan_balanced_clusters.geojson', driver='GeoJSON')
print("\nSaved manhattan_balanced_clusters.geojson")

# Map
print("Creating map...")
manhattan_center = [40.7831, -73.9712]
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
          '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
          '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
          '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
          '#ff6666', '#66ff66', '#6666ff', '#ffff66', '#ff66ff']

def get_color(c):
    if c < 0: return '#333333'
    return colors[int(c) % len(colors)]

m = folium.Map(location=manhattan_center, zoom_start=12, tiles='cartodbdark_matter')
folium.GeoJson(
    result,
    style_function=lambda f: {
        'fillColor': get_color(f['properties'].get('cluster')),
        'color': '#000000', 'weight': 0.5, 'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['cluster', 'avg_year', 'avg_value_m', 'res_pct',
                'airbnb_total', 'schools', 'avg_sat', 'crimes', 'subway_m'],
        aliases=['Cluster', 'Avg Year Built', 'Avg Value ($M)', 'Residential %',
                 'Airbnb Listings', 'Schools', 'Avg SAT', 'Crimes', 'Subway Dist (m)'],
        localize=True
    )
).add_to(m)

m.save('../out/html/manhattan_balanced.html')
print("Saved manhattan_balanced.html")
