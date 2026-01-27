"""Quick Manhattan visualization"""
import pandas as pd
import geopandas as gpd
import folium

gdf = gpd.read_file('../data/clean/manhattan_cluster_results.geojson')
print(f"Columns: {gdf.columns.tolist()}")

# Find correct column names
ward_col = [c for c in gdf.columns if 'ward_cluster' in c][0]
kmeans_col = [c for c in gdf.columns if 'kmeans_cluster' in c][0]
print(f"Using: {ward_col}, {kmeans_col}")

manhattan_center = [40.7831, -73.9712]
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
          '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']

def get_color(cluster):
    if pd.isna(cluster):
        return '#333333'
    return colors[int(cluster) % len(colors)]

m = folium.Map(location=manhattan_center, zoom_start=12, tiles='cartodbdark_matter')

folium.GeoJson(
    gdf,
    style_function=lambda feature: {
        'fillColor': get_color(feature['properties'].get(ward_col)),
        'color': '#000000',
        'weight': 0.3,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['h3_10', ward_col, kmeans_col],
        aliases=['H3 ID', 'Ward Cluster', 'KMeans Cluster'],
        localize=True
    )
).add_to(m)

m.save('../out/html/manhattan_clusters.html')
print("Saved manhattan_clusters.html")
