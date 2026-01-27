"""
Run clustering on Manhattan with more clusters and POI data
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import h3pandas
from h3 import cell_to_latlng
from libpysal.weights import Queen
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import robust_scale
import folium
import warnings
warnings.filterwarnings('ignore')

# Load Manhattan hex list
print("Loading Manhattan hexagons...")
manhattan_hexes = pd.read_csv('../data/clean/h3_index/manhattan_hex.csv')['manhattan_hex'].tolist()
print(f"Manhattan has {len(manhattan_hexes)} hexagons in index")

# Load PLUTO data
print("Loading h3_PLUTO.csv...")
df = pd.read_csv('../data/clean/h3_PLUTO.csv', low_memory=False)
df = df[df['h3_10'].isin(manhattan_hexes)]
print(f"Filtered to {len(df)} Manhattan PLUTO records")

# Aggregate PLUTO by H3 hexagon
print("Aggregating PLUTO by H3 hexagon...")
pluto_agg = df.groupby('h3_10').agg({
    'BldgArea': 'sum',
    'YearBuilt': 'mean',
    'AssessTot': 'sum',  # Changed to sum for total assessed value
    'ResArea': 'sum',
    'ComArea': 'sum',     # Commercial area
    'OfficeArea': 'sum',  # Office area
    'RetailArea': 'sum',  # Retail area
    'NumBldgs': 'sum',    # Number of buildings
    'UnitsRes': 'sum',    # Residential units
}).reset_index()

# Filter out invalid years
pluto_agg = pluto_agg[pluto_agg['YearBuilt'] > 1500]

# Load POI data
print("Loading POI data...")
poi_df = pd.read_csv('../data/clean/poi_h3.csv')
poi_df = poi_df[poi_df['h3_10'].isin(manhattan_hexes)]

# Count POIs per hexagon
poi_counts = poi_df.groupby('h3_10').size().reset_index(name='poi_count')

# Merge PLUTO with POI counts
df_gb = pluto_agg.merge(poi_counts, on='h3_10', how='left')
df_gb['poi_count'] = df_gb['poi_count'].fillna(0)

# Add lat/lon from H3
df_gb['lat'], df_gb['lon'] = zip(*df_gb['h3_10'].apply(cell_to_latlng))

print(f"Aggregated to {len(df_gb)} hexagons with {df_gb['poi_count'].sum():.0f} total POIs")

# Cluster variables - expanded feature set
cluster_variables = [
    'BldgArea',      # Total building area
    'YearBuilt',     # Average year built
    'AssessTot',     # Total assessed value
    'ResArea',       # Residential area
    'ComArea',       # Commercial area
    'RetailArea',    # Retail area
    'UnitsRes',      # Residential units
    'poi_count',     # Points of interest
]

# K-Means clustering (15 clusters)
print("Running K-Means clustering (15 clusters)...")
X_scaled = robust_scale(df_gb[cluster_variables].values)
kmeans = KMeans(n_clusters=15, init='k-means++', random_state=42, n_init=10)
df_gb['kmeans_cluster'] = kmeans.fit_predict(X_scaled) + 1

# Create GeoDataFrame with H3 boundaries
print("Creating spatial weights matrix...")
df_map = df_gb.set_index('h3_10')
gdf = df_map.h3.h3_to_geo_boundary().reset_index()

# Build Queen contiguity weights
w = Queen.from_dataframe(gdf)
print(f"Found {len(w.islands)} island hexagons")

# Remove islands for spatial clustering
clean = gdf.drop(w.islands).copy()
w2 = Queen.from_dataframe(clean)

# Scale data for clustering
db_scaled = robust_scale(clean[cluster_variables])

# Ward clustering with spatial constraint (25 clusters for Manhattan)
print("Running Ward clustering with spatial constraint (25 clusters)...")
ward_model = AgglomerativeClustering(
    linkage='ward',
    connectivity=w2.sparse,
    n_clusters=25
)
clean['ward_cluster'] = ward_model.fit_predict(db_scaled)

# Merge cluster results back
print("Merging results...")
result = df_gb.copy()
cluster_map = clean.set_index('h3_10')[['ward_cluster']]
result = result.set_index('h3_10').join(cluster_map).reset_index()

# Create geometry
result_gdf = result.set_index('h3_10').h3.h3_to_geo_boundary().reset_index()

# Export
print("Exporting results...")
result.to_csv('../data/clean/manhattan_cluster_results_v2.csv', index=False)
print(f"Saved CSV ({len(result)} hexagons)")

geojson_gdf = gpd.GeoDataFrame(result_gdf.merge(result[['h3_10', 'kmeans_cluster', 'ward_cluster']], on='h3_10'),
                               geometry='geometry', crs='EPSG:4326')
geojson_gdf.to_file('../data/clean/manhattan_cluster_results_v2.geojson', driver='GeoJSON')
print("Saved GeoJSON")

# Visualization
print("\nCreating map...")
manhattan_center = [40.7831, -73.9712]
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
          '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
          '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
          '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
          '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff']

def get_color(cluster):
    if pd.isna(cluster):
        return '#333333'
    return colors[int(cluster) % len(colors)]

m = folium.Map(location=manhattan_center, zoom_start=13, tiles='cartodbdark_matter')

# Use ward_cluster for coloring
ward_col = 'ward_cluster'
folium.GeoJson(
    geojson_gdf,
    style_function=lambda feature: {
        'fillColor': get_color(feature['properties'].get(ward_col)),
        'color': '#000000',
        'weight': 0.3,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['h3_10', 'ward_cluster', 'kmeans_cluster'],
        aliases=['H3 ID', 'Ward Cluster', 'KMeans Cluster'],
        localize=True
    )
).add_to(m)

m.save('../out/html/manhattan_clusters_v2.html')
print("Saved manhattan_clusters_v2.html")

# Summary
print("\n--- Clustering Summary ---")
print(f"Total hexagons: {len(result)}")
print(f"Features used: {cluster_variables}")
print(f"\nWard spatial clusters (25):")
for cluster_id in sorted(result['ward_cluster'].dropna().unique()):
    count = (result['ward_cluster'] == cluster_id).sum()
    print(f"  Cluster {int(cluster_id)}: {count} hexagons")
