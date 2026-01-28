"""
Process Airbnb and School SAT data and aggregate to H3 hexagons
"""

import pandas as pd
import numpy as np
from h3 import latlng_to_cell, cell_to_latlng

H3_RESOLUTION = 10

# Load Manhattan hexagons
print("Loading Manhattan hexagons...")
manhattan_hexes = pd.read_csv('../data/clean/h3_index/manhattan_hex.csv')['manhattan_hex'].tolist()

# =============================================================================
# 1. AIRBNB LISTINGS
# =============================================================================
print("\n=== Processing Airbnb Listings ===")
airbnb_df = pd.read_csv('../data/raw/external/airbnb_listings.csv')

# Filter to Manhattan only
airbnb_manhattan = airbnb_df[airbnb_df['neighbourhood_group'] == 'Manhattan'].copy()
print(f"Manhattan has {len(airbnb_manhattan)} Airbnb listings")

# Assign H3 index
airbnb_manhattan['h3_10'] = airbnb_manhattan.apply(
    lambda r: latlng_to_cell(r['latitude'], r['longitude'], H3_RESOLUTION), axis=1
)

# Aggregate by hex
airbnb_agg = airbnb_manhattan.groupby('h3_10').agg({
    'id': 'count',  # number of listings
    'price': 'median',  # median price
    'number_of_reviews': 'sum',  # total reviews (proxy for activity)
    'availability_365': 'mean',  # avg availability
}).reset_index()

airbnb_agg.columns = ['h3_10', 'airbnb_count', 'airbnb_price_median',
                      'airbnb_reviews_total', 'airbnb_availability_avg']

# Also get room type breakdown
room_type_pivot = pd.crosstab(airbnb_manhattan['h3_10'], airbnb_manhattan['room_type'])
room_type_pivot = room_type_pivot.reset_index()
room_type_pivot.columns = ['h3_10'] + [f'airbnb_{c.lower().replace(" ", "_").replace("/", "_")}'
                                        for c in room_type_pivot.columns[1:]]

airbnb_features = airbnb_agg.merge(room_type_pivot, on='h3_10', how='left')

print(f"Airbnb data for {len(airbnb_features)} hexagons")
print(f"Total listings: {airbnb_features['airbnb_count'].sum()}")

# =============================================================================
# 2. SCHOOL SAT SCORES
# =============================================================================
print("\n=== Processing School SAT Scores ===")
schools_df = pd.read_csv('../data/raw/external/school_sat_scores.csv')

# Filter to Manhattan only (Borough column)
schools_manhattan = schools_df[schools_df['Borough'] == 'Manhattan'].copy()
schools_manhattan = schools_manhattan.dropna(subset=['Latitude', 'Longitude'])
print(f"Manhattan has {len(schools_manhattan)} high schools with location data")

# Assign H3 index
schools_manhattan['h3_10'] = schools_manhattan.apply(
    lambda r: latlng_to_cell(r['Latitude'], r['Longitude'], H3_RESOLUTION), axis=1
)

# Clean numeric columns
def parse_pct(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        return float(x.replace('%', '')) / 100
    return x

for col in ['Percent White', 'Percent Black', 'Percent Hispanic', 'Percent Asian', 'Percent Tested']:
    schools_manhattan[col] = schools_manhattan[col].apply(parse_pct)

# Aggregate by hex
schools_agg = schools_manhattan.groupby('h3_10').agg({
    'School ID': 'count',  # number of schools
    'Student Enrollment': 'sum',  # total students
    'Average Score (SAT Math)': 'mean',
    'Average Score (SAT Reading)': 'mean',
    'Average Score (SAT Writing)': 'mean',
}).reset_index()

schools_agg.columns = ['h3_10', 'school_count', 'school_enrollment_total',
                       'school_sat_math_avg', 'school_sat_reading_avg', 'school_sat_writing_avg']

print(f"School data for {len(schools_agg)} hexagons")
print(f"Total schools: {schools_agg['school_count'].sum()}")

# =============================================================================
# MERGE WITH EXISTING EXTERNAL FEATURES
# =============================================================================
print("\n=== Merging with existing external features ===")

# Load existing external features
existing_features = pd.read_csv('../data/clean/external_features_h3.csv')
print(f"Existing features: {len(existing_features)} hexagons, {len(existing_features.columns)} columns")

# Merge Airbnb
all_features = existing_features.merge(airbnb_features, on='h3_10', how='left')

# Merge schools
all_features = all_features.merge(schools_agg, on='h3_10', how='left')

# Fill NaN with 0 for count columns
count_cols = ['airbnb_count', 'airbnb_reviews_total', 'school_count', 'school_enrollment_total']
for col in count_cols:
    if col in all_features.columns:
        all_features[col] = all_features[col].fillna(0)

# Fill NaN for room type columns
room_type_cols = [c for c in all_features.columns if c.startswith('airbnb_') and c not in
                  ['airbnb_count', 'airbnb_price_median', 'airbnb_reviews_total', 'airbnb_availability_avg']]
for col in room_type_cols:
    if col in all_features.columns:
        all_features[col] = all_features[col].fillna(0)

# Save
output_path = '../data/clean/external_features_h3.csv'
all_features.to_csv(output_path, index=False)
print(f"\nSaved {len(all_features)} hexagons with {len(all_features.columns)} features to {output_path}")

# Summary
print("\n=== Feature Summary ===")
print("New columns added:")
for col in all_features.columns:
    if 'airbnb' in col or 'school' in col:
        non_null = all_features[col].notna().sum()
        print(f"  {col}: {non_null} non-null values")
