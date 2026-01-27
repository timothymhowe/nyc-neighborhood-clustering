"""
Regenerate h3_PLUTO.csv from MapPLUTO shapefile.
Based on src/h3_indexing-BBL.ipynb
"""

import pandas as pd
import geopandas as gpd
import h3pandas

print("Loading MapPLUTO shapefile...")
gdf_in = gpd.read_file("../data/nyc_mappluto_25v3_shp/MapPLUTO.shp")
print(f"Loaded {len(gdf_in)} records")

# H3 resolution 10 (same as original)
parent_resolution = 10

# Clean data
gdf = gdf_in.copy()
gdf = gdf.dropna(subset='BCTCB2020')
gdf['BCTCB2020'] = gdf['BCTCB2020'].astype(int)
gdf['BBL'] = gdf['BBL'].astype(int)

print(f"After cleaning: {len(gdf)} records")

# Create H3 index from lat/lon
h3_df = gdf[['BBL', 'BCTCB2020', 'Latitude', 'Longitude']].copy()
h3_df = h3_df.rename({'Latitude': 'lat', 'Longitude': 'lon'}, axis=1)

print("Converting to H3 index...")
h3_df = h3_df.h3.geo_to_h3(parent_resolution, lat_col='lat', lng_col='lon')

out_df = h3_df.reset_index().set_index('BBL')[['h3_10']]

# Save BBL to H3 mapping
print("Saving bbl_to_h3.json...")
out_df.to_json('../data/clean/bbl_to_h3.json', orient='index')

# Merge back with full PLUTO data
print("Merging with full PLUTO data...")
out = gdf_in.merge(out_df, on='BBL')

# Save full h3_PLUTO.csv
print("Saving h3_PLUTO.csv...")
out.to_csv('../data/clean/h3_PLUTO.csv', index=False)

print(f"Done! Saved {len(out)} records to h3_PLUTO.csv")
print(f"Unique H3 hexagons: {out['h3_10'].nunique()}")
