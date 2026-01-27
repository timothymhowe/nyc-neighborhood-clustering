# NYC Neighborhood Clustering Methodology

## Overview
This project clusters NYC tax lots (aggregated to H3 hexagons) into contiguous neighborhood-like regions based on built environment characteristics.

## Data Sources

### Current
1. **MapPLUTO** (NYC Dept of City Planning, v25.3)
   - Tax lot level data: building area, year built, assessed value, land use
   - ~856k tax lots aggregated to H3 resolution 10 hexagons

2. **NYC Points of Interest** (DoITT)
   - Government facilities, parks, schools, etc.
   - Count of POIs per hexagon as a feature

### Planned Additions
- Foursquare/Yelp venue data (restaurants, bars, retail categories)
- NYPD crime data
- 311 complaint data
- Subway accessibility metrics

## Spatial Unit
- **H3 Hexagonal Grid** at resolution 10
- Each hexagon is ~15,000 m² (roughly 1.5 city blocks)
- Hexagons provide uniform neighbor relationships (6 neighbors each)

## Feature Engineering

Features aggregated per hexagon:
| Feature | Aggregation | Description |
|---------|-------------|-------------|
| BldgArea | sum | Total building floor area |
| YearBuilt | mean | Average construction year |
| AssessTot | sum | Total assessed property value |
| ResArea | sum | Total residential floor area |
| ComArea | sum | Total commercial floor area |
| RetailArea | sum | Total retail floor area |
| UnitsRes | sum | Total residential units |
| poi_count | count | Number of points of interest |

## Algorithm Selection

### Why SKATER?
We use **SKATER** (Spatial 'K'luster Analysis by Tree Edge Removal) from the `spopt` library.

**Problem with standard clustering (K-Means, Ward):**
- Does not guarantee spatial contiguity
- Clusters can be fragmented across non-adjacent areas
- Even with spatial weights in Ward clustering, contiguity is soft, not enforced

**SKATER approach:**
1. Build a minimum spanning tree from the spatial weights graph
2. Prune edges to create k regions
3. **Guarantees contiguous regions** by construction

### Algorithm Parameters
- `n_clusters=25` - Target number of neighborhoods for Manhattan
- `floor=5` - Minimum hexagons per cluster (prevents tiny fragments)
- Spatial weights: Queen contiguity (shared edges or vertices)

### Preprocessing Steps
1. Filter to largest connected component (removes ~64 island hexagons in waterways)
2. Apply `robust_scale` to features (handles outliers in assessed values)
3. Remove hexagons with invalid YearBuilt (< 1500)

## Version History

### v1 (Initial)
- K-Means and Ward clustering
- 4-5 clusters, heavily imbalanced
- No contiguity guarantee

### v2 (Current)
- SKATER regionalization
- 25 contiguous clusters
- Expanded feature set (8 variables)
- Verified 100% contiguity

## Validation
Contiguity is verified programmatically using BFS traversal within each cluster to ensure all members are reachable through neighbor relationships.

## Files
- `run_clustering_contiguous.py` - Main clustering script
- `data/clean/manhattan_contiguous_clusters.geojson` - Output with cluster assignments
- `out/html/manhattan_contiguous.html` - Interactive visualization
