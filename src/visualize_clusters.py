"""
Create an interactive map visualization of cluster results
"""

import pandas as pd
import geopandas as gpd
import folium

print("Loading cluster results...")
gdf = gpd.read_file('../data/clean/cluster_results.geojson')

# NYC center
nyc_center = [40.7128, -74.0060]

# Create map
print("Creating map...")
m = folium.Map(location=nyc_center, zoom_start=11, tiles='cartodbdark_matter')

# Color palette for clusters
colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
]

def get_color(cluster):
    if pd.isna(cluster):
        return '#333333'
    return colors[int(cluster) % len(colors)]

# Find the correct column name (might have _x or _y suffix from merge)
ward_col = [c for c in gdf.columns if 'ward_cluster' in c][0]
kmeans_col = [c for c in gdf.columns if 'kmeans_cluster' in c][0]

print(f"Using columns: {ward_col}, {kmeans_col}")

# Add hexagons colored by ward_cluster (spatially constrained)
print("Adding hexagons...")
folium.GeoJson(
    gdf,
    style_function=lambda feature: {
        'fillColor': get_color(feature['properties'].get(ward_col)),
        'color': '#000000',
        'weight': 0.3,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['h3_10', ward_col, kmeans_col, 'BldgArea', 'YearBuilt'],
        aliases=['H3 ID', 'Ward Cluster', 'KMeans Cluster', 'Building Area', 'Avg Year Built'],
        localize=True
    )
).add_to(m)

# Save
output_path = '../out/html/cluster_map.html'
m.save(output_path)
print(f"Saved to {output_path}")
print("Open this file in your browser to view the map!")
