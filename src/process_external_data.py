"""
Process external data sources and aggregate to H3 hexagons
- Subway stations: distance to nearest station per hexagon
- NYPD complaints: crime counts by category
- 311 complaints: complaint counts by type
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from h3 import latlng_to_cell, cell_to_latlng, great_circle_distance
import h3pandas
from scipy.spatial import cKDTree

H3_RESOLUTION = 10

# Load Manhattan hexagons
print("Loading Manhattan hexagons...")
manhattan_hexes = pd.read_csv('../data/clean/h3_index/manhattan_hex.csv')['manhattan_hex'].tolist()

# =============================================================================
# 1. SUBWAY STATIONS - Distance to nearest station
# =============================================================================
print("\n=== Processing Subway Stations ===")
subway_df = pd.read_csv('../data/raw/external/subway_stations.csv')

# Filter to Manhattan
subway_manhattan = subway_df[subway_df['Borough'] == 'M'].copy()
print(f"Manhattan has {len(subway_manhattan)} subway stations")

# Get station coordinates
station_coords = subway_manhattan[['GTFS Latitude', 'GTFS Longitude']].values

# Build KD-tree for fast nearest neighbor search
station_tree = cKDTree(np.radians(station_coords))

# For each hex, compute distance to nearest station
def get_hex_center(h3_id):
    lat, lng = cell_to_latlng(h3_id)
    return lat, lng

hex_centers = [get_hex_center(h) for h in manhattan_hexes]
hex_coords_rad = np.radians(hex_centers)

# Query nearest station for each hex
distances, indices = station_tree.query(hex_coords_rad, k=1)

# Convert angular distance to meters (approximate)
R_EARTH = 6371000  # Earth radius in meters
distances_m = distances * R_EARTH

subway_features = pd.DataFrame({
    'h3_10': manhattan_hexes,
    'subway_dist_m': distances_m,
    'nearest_station': subway_manhattan.iloc[indices]['Stop Name'].values
})

print(f"Avg distance to subway: {subway_features['subway_dist_m'].mean():.0f}m")
print(f"Max distance to subway: {subway_features['subway_dist_m'].max():.0f}m")

# =============================================================================
# 2. NYPD COMPLAINTS - Crime counts by category
# =============================================================================
print("\n=== Processing NYPD Complaints ===")
nypd_df = pd.read_csv('../data/raw/external/nypd_complaints_manhattan_2023.csv', low_memory=False)

# Drop rows without coordinates
nypd_df = nypd_df.dropna(subset=['latitude', 'longitude'])
print(f"Loaded {len(nypd_df)} complaints with coordinates")

# Assign H3 index
nypd_df['h3_10'] = nypd_df.apply(
    lambda r: latlng_to_cell(r['latitude'], r['longitude'], H3_RESOLUTION), axis=1
)

# Aggregate by hex
crime_counts = nypd_df.groupby('h3_10').size().reset_index(name='crime_count')

# Also get breakdown by law category
law_cat_pivot = pd.crosstab(nypd_df['h3_10'], nypd_df['law_cat_cd'])
law_cat_pivot = law_cat_pivot.reset_index()
law_cat_pivot.columns = ['h3_10'] + [f'crime_{c.lower()}' for c in law_cat_pivot.columns[1:]]

crime_features = crime_counts.merge(law_cat_pivot, on='h3_10', how='left')
print(f"Crime data for {len(crime_features)} hexagons")
print(f"Total crimes: {crime_features['crime_count'].sum()}")

# =============================================================================
# 3. 311 COMPLAINTS - Quality of life indicators
# =============================================================================
print("\n=== Processing 311 Complaints ===")
complaints_df = pd.read_csv('../data/raw/external/311_manhattan.csv', low_memory=False)

# Drop rows without coordinates
complaints_df = complaints_df.dropna(subset=['latitude', 'longitude'])
print(f"Loaded {len(complaints_df)} 311 complaints with coordinates")

# Assign H3 index
complaints_df['h3_10'] = complaints_df.apply(
    lambda r: latlng_to_cell(r['latitude'], r['longitude'], H3_RESOLUTION), axis=1
)

# Total 311 count
complaint_counts = complaints_df.groupby('h3_10').size().reset_index(name='complaint_311_count')

# Key complaint categories
key_types = ['Noise - Residential', 'Noise - Commercial', 'Illegal Parking',
             'HEAT/HOT WATER', 'Street Condition', 'Blocked Driveway']

for ctype in key_types:
    col_name = f"c311_{ctype.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}"
    type_counts = complaints_df[complaints_df['complaint_type'] == ctype].groupby('h3_10').size()
    complaint_counts[col_name] = complaint_counts['h3_10'].map(type_counts).fillna(0)

print(f"311 data for {len(complaint_counts)} hexagons")
print(f"Total 311 complaints: {complaint_counts['complaint_311_count'].sum()}")

# =============================================================================
# MERGE ALL FEATURES
# =============================================================================
print("\n=== Merging Features ===")

# Start with all Manhattan hexes
all_features = pd.DataFrame({'h3_10': manhattan_hexes})

# Merge subway features
all_features = all_features.merge(subway_features[['h3_10', 'subway_dist_m']], on='h3_10', how='left')

# Merge crime features
all_features = all_features.merge(crime_features, on='h3_10', how='left')

# Merge 311 features
all_features = all_features.merge(complaint_counts, on='h3_10', how='left')

# Fill NaN with 0 for count columns
count_cols = [c for c in all_features.columns if 'count' in c or 'crime_' in c or 'c311_' in c]
all_features[count_cols] = all_features[count_cols].fillna(0)

# Save
output_path = '../data/clean/external_features_h3.csv'
all_features.to_csv(output_path, index=False)
print(f"\nSaved {len(all_features)} hexagons with external features to {output_path}")

# Summary
print("\n=== Feature Summary ===")
print(all_features.describe())
