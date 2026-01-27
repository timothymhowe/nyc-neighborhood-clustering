"""
Run clustering on H3-indexed PLUTO data - Manhattan only
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
print(f"Manhattan has {len(manhattan_hexes)} hexagons")

print("Loading h3_PLUTO.csv...")
df = pd.read_csv('../data/clean/h3_PLUTO.csv', low_memory=False)
print(f"Loaded {len(df)} records")

# Filter to Manhattan only
df = df[df['h3_10'].isin(manhattan_hexes)]
print(f"Filtered to {len(df)} Manhattan records")

# Aggregate by H3 hexagon
print("Aggregating by H3 hexagon...")
df_gb = df.groupby('h3_10').agg({
    'BldgArea': 'sum',
    'YearBuilt': 'mean',
    'AssessTot': 'max',
    'ResArea': 'sum'
})

# Filter out invalid years
df_gb = df_gb[df_gb['YearBuilt'] > 1500]
df_gb = df_gb.reset_index()

# Add lat/lon from H3
df2 = pd.DataFrame(df_gb['h3_10'])
df_gb['lat'], df_gb['lon'] = zip(*df2['h3_10'].apply(cell_to_latlng))

print(f"Aggregated to {len(df_gb)} hexagons")

# Cluster variables
cluster_variables = ['BldgArea', 'YearBuilt', 'AssessTot', 'ResArea']

# K-Means clustering (8 clusters) - use scaled data
print("Running K-Means clustering (8 clusters)...")
X_scaled = robust_scale(df_gb[cluster_variables].values)
kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)
df_gb['kmeans_cluster'] = y_kmeans + 1

# Create GeoDataFrame with H3 boundaries
print("Creating spatial weights matrix...")
df_map = df_gb.set_index('h3_10')
gdf = df_map.h3.h3_to_geo_boundary().reset_index()

# Build Queen contiguity weights
w = Queen.from_dataframe(gdf)
print(f"Found {len(w.islands)} island hexagons")

# Remove islands for spatial clustering
clean = gdf.drop(w.islands).copy()

# Rebuild weights without islands
w2 = Queen.from_dataframe(clean)

# Scale data for clustering
db_scaled = robust_scale(clean[cluster_variables])

# Agglomerative clustering (12 clusters, no spatial constraint)
print("Running Agglomerative clustering (12 clusters)...")
agg = AgglomerativeClustering(n_clusters=12)
clean['agg_cluster'] = agg.fit_predict(db_scaled)

# Ward clustering with spatial constraint (10 clusters for Manhattan)
print("Running Ward clustering with spatial constraint (10 clusters)...")
ward_model = AgglomerativeClustering(
    linkage='ward',
    connectivity=w2.sparse,
    n_clusters=10
)
clean['ward_cluster'] = ward_model.fit_predict(db_scaled)

# Merge cluster results back to full dataset
print("Merging results...")
result = df_gb.copy()
cluster_map = clean.set_index('h3_10')[['agg_cluster', 'ward_cluster']]
result = result.set_index('h3_10').join(cluster_map).reset_index()

# Create geometry for GeoJSON export
print("Creating H3 boundaries...")
result_gdf = result.set_index('h3_10').h3.h3_to_geo_boundary().reset_index()

# Export results
print("Exporting results...")

# CSV with cluster assignments
csv_output = result[['h3_10', 'lat', 'lon', 'BldgArea', 'YearBuilt', 'AssessTot', 'ResArea',
                     'kmeans_cluster', 'agg_cluster', 'ward_cluster']].copy()
csv_output.to_csv('../data/clean/manhattan_cluster_results.csv', index=False)
print(f"Saved manhattan_cluster_results.csv ({len(csv_output)} hexagons)")

# GeoJSON for frontend
geojson_gdf = result_gdf.merge(result[['h3_10', 'kmeans_cluster', 'agg_cluster', 'ward_cluster']], on='h3_10')
geojson_gdf = gpd.GeoDataFrame(geojson_gdf, geometry='geometry', crs='EPSG:4326')
geojson_gdf.to_file('../data/clean/manhattan_cluster_results.geojson', driver='GeoJSON')
print("Saved manhattan_cluster_results.geojson")

# Create visualization
print("\nCreating map visualization...")
manhattan_center = [40.7831, -73.9712]

colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'
]

def get_color(cluster):
    if pd.isna(cluster):
        return '#333333'
    return colors[int(cluster) % len(colors)]

m = folium.Map(location=manhattan_center, zoom_start=12, tiles='cartodbdark_matter')

folium.GeoJson(
    geojson_gdf,
    style_function=lambda feature: {
        'fillColor': get_color(feature['properties'].get('ward_cluster')),
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

m.save('../out/html/manhattan_clusters.html')
print("Saved manhattan_clusters.html")

# Summary stats
print("\n--- Manhattan Clustering Summary ---")
print(f"Total hexagons: {len(result)}")
print(f"K-Means clusters (8): {result['kmeans_cluster'].value_counts().sort_index().to_dict()}")
print(f"Ward spatial clusters (10): {result['ward_cluster'].value_counts().sort_index().to_dict()}")
print("\nDone!")
