"""
Run clustering on Manhattan with GUARANTEED contiguous clusters using SKATER
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import h3pandas
from h3 import cell_to_latlng
from libpysal.weights import Queen
from sklearn.preprocessing import robust_scale
from spopt.region import Skater
import folium
import warnings
warnings.filterwarnings('ignore')

# Load Manhattan hex list
print("Loading Manhattan hexagons...")
manhattan_hexes = pd.read_csv('../data/clean/h3_index/manhattan_hex.csv')['manhattan_hex'].tolist()

# Load PLUTO data
print("Loading h3_PLUTO.csv...")
df = pd.read_csv('../data/clean/h3_PLUTO.csv', low_memory=False)
df = df[df['h3_10'].isin(manhattan_hexes)]

# Aggregate PLUTO by H3 hexagon
print("Aggregating PLUTO by H3 hexagon...")
pluto_agg = df.groupby('h3_10').agg({
    'BldgArea': 'sum',
    'YearBuilt': 'mean',
    'AssessTot': 'sum',
    'ResArea': 'sum',
    'ComArea': 'sum',
    'RetailArea': 'sum',
    'UnitsRes': 'sum',
}).reset_index()

pluto_agg = pluto_agg[pluto_agg['YearBuilt'] > 1500]

# Load POI data
print("Loading POI data...")
poi_df = pd.read_csv('../data/clean/poi_h3.csv')
poi_df = poi_df[poi_df['h3_10'].isin(manhattan_hexes)]
poi_counts = poi_df.groupby('h3_10').size().reset_index(name='poi_count')

# Merge POI counts
df_gb = pluto_agg.merge(poi_counts, on='h3_10', how='left')
df_gb['poi_count'] = df_gb['poi_count'].fillna(0)
df_gb['lat'], df_gb['lon'] = zip(*df_gb['h3_10'].apply(cell_to_latlng))

# Load and merge external features (subway, crime, 311)
print("Loading external features...")
external_df = pd.read_csv('../data/clean/external_features_h3.csv')
df_gb = df_gb.merge(external_df, on='h3_10', how='left')

# Fill NaN for external features
external_cols = ['subway_dist_m', 'crime_count', 'complaint_311_count']
for col in external_cols:
    if col in df_gb.columns:
        df_gb[col] = df_gb[col].fillna(df_gb[col].median())

print(f"Aggregated to {len(df_gb)} hexagons")

# Cluster variables - expanded with external data
cluster_variables = [
    # Built environment (PLUTO)
    'BldgArea', 'YearBuilt', 'AssessTot', 'ResArea',
    'ComArea', 'RetailArea', 'UnitsRes',
    # POI
    'poi_count',
    # External data
    'subway_dist_m',     # Transit accessibility
    'crime_count',       # Safety/crime
    'complaint_311_count',  # Quality of life
]

# Create GeoDataFrame with H3 boundaries
print("Creating GeoDataFrame...")
gdf = df_gb.set_index('h3_10').h3.h3_to_geo_boundary().reset_index()
gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')

# Build Queen contiguity weights
print("Building spatial weights...")
w = Queen.from_dataframe(gdf)
print(f"Found {len(w.islands)} island hexagons")

# Remove islands (SKATER requires connected graph)
# Also need to find the largest connected component
import networkx as nx

print("Finding largest connected component...")
G = w.to_networkx()
components = list(nx.connected_components(G))
print(f"Found {len(components)} connected components")

# Get the largest component
largest_component = max(components, key=len)
print(f"Largest component has {len(largest_component)} hexagons")

# Filter to only the largest connected component
gdf_clean = gdf.iloc[list(largest_component)].reset_index(drop=True)
w_clean = Queen.from_dataframe(gdf_clean)

print(f"Clustering {len(gdf_clean)} hexagons...")

# Scale features
X = gdf_clean[cluster_variables].values
X_scaled = robust_scale(X)

# SKATER - guarantees contiguous regions
# n_clusters = 17 for Manhattan neighborhoods
N_CLUSTERS = 17
print(f"Running SKATER regionalization ({N_CLUSTERS} contiguous clusters)...")

skater = Skater(
    gdf_clean,
    w_clean,
    cluster_variables,
    n_clusters=N_CLUSTERS,
    floor=5,  # Minimum hexagons per cluster
    trace=False,
    islands='ignore'
)
skater.solve()

gdf_clean['cluster'] = skater.labels_

print(f"Created {gdf_clean['cluster'].nunique()} contiguous clusters")

# Verify contiguity
print("\nVerifying contiguity...")
from collections import defaultdict
neighbors = defaultdict(set)
for i, neighs in w_clean.neighbors.items():
    for n in neighs:
        neighbors[i].add(n)

def check_contiguity(labels, neighbors):
    """Check if each cluster forms a contiguous region"""
    from collections import deque

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)

    non_contiguous = []
    for cluster_id, members in clusters.items():
        if len(members) <= 1:
            continue

        # BFS to check connectivity within cluster
        visited = set()
        queue = deque([members[0]])
        visited.add(members[0])

        while queue:
            node = queue.popleft()
            for neighbor in neighbors[node]:
                if neighbor in members and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(visited) != len(members):
            non_contiguous.append(cluster_id)

    return non_contiguous

non_contig = check_contiguity(gdf_clean['cluster'].values, neighbors)
if non_contig:
    print(f"WARNING: {len(non_contig)} clusters are not contiguous: {non_contig}")
else:
    print("All clusters are contiguous!")

# Add back islands with cluster = -1
result = gdf.merge(gdf_clean[['h3_10', 'cluster']], on='h3_10', how='left')
result['cluster'] = result['cluster'].fillna(-1).astype(int)

# Export
print("\nExporting results...")
result.to_file('../data/clean/manhattan_contiguous_clusters.geojson', driver='GeoJSON')
print("Saved GeoJSON")

# Visualization
print("Creating map...")
manhattan_center = [40.7831, -73.9712]
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
          '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
          '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
          '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
          '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff']

def get_color(cluster):
    if cluster == -1 or pd.isna(cluster):
        return '#333333'
    return colors[int(cluster) % len(colors)]

m = folium.Map(location=manhattan_center, zoom_start=13, tiles='cartodbdark_matter')

folium.GeoJson(
    result,
    style_function=lambda feature: {
        'fillColor': get_color(feature['properties'].get('cluster')),
        'color': '#000000',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['h3_10', 'cluster'],
        aliases=['H3 ID', 'Cluster'],
    )
).add_to(m)

m.save('../out/html/manhattan_contiguous.html')
print("Saved manhattan_contiguous.html")

# Summary
print("\n--- Clustering Summary ---")
print(f"Total hexagons: {len(result)}")
print(f"Clustered hexagons: {(result['cluster'] >= 0).sum()}")
print(f"Island hexagons: {(result['cluster'] == -1).sum()}")
print(f"\nCluster sizes:")
for c in sorted(result['cluster'].unique()):
    if c >= 0:
        count = (result['cluster'] == c).sum()
        print(f"  Cluster {c}: {count} hexagons")
