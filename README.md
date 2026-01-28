# the manhattan project

*a machine learning re-division of manhattan*

**[Live Demo](https://nyc-neighborhood-clustering.vercel.app)**

## what is this

Manhattan's neighborhood boundaries are basically vibes. Tribeca, SoHo, NoHo - these names came from real estate agents and newspaper columnists, not data. This project asks: what if we let the actual characteristics of the city define the boundaries? What would Manhattan look like if we clustered by buildings, transit access, crime, schools, and property values instead of tradition?

The result is 34 contiguous "neighborhoods" that emerge purely from the data. Some match intuition (Midtown is its own thing, obviously). Others are weirder - like how the Upper West Side fragments into distinct micro-zones based on building age and density.

## the approach

**Spatial indexing**: Every property, subway station, school, crime report, and Airbnb listing gets mapped to Uber's H3 hexagonal grid at resolution 10 (~0.015 km² per cell). This gives us a consistent unit of analysis across all data sources.

**Clustering**: SKATER (Spatial 'K'luster Analysis by Tree Edge Removal) from the PySAL ecosystem. Unlike k-means, SKATER respects spatial contiguity - clusters can't have random disconnected pieces. It builds a minimum spanning tree on the Queen contiguity graph and prunes edges to create regions.

**Features used**:
- Building characteristics (area, year built, assessed value, residential vs commercial)
- Price per square foot (residential and commercial, adjusted from assessed to market value)
- Transit access (distance to nearest subway)
- Activity indicators (crime density, 311 complaints, Airbnb concentration)
- Schools (count and enrollment)

## the gnarly parts

A few things that weren't obvious:

**Assessment ratios**: NYC's assessed values aren't market values. Residential is assessed at ~4% of market, commercial at ~45%. Took some digging through DOF documentation to figure out why my $/sqft numbers were coming out at $60 instead of $1,500.

**Data holes**: Some hexes had buildings but no YearBuilt data (or years like "1800" for clearly modern buildings). Instead of dropping them and creating gaps in the map, imputed with the median (1918) for invalid values.

**Contiguity optimization**: Ran a grid search across cluster parameters to minimize fragmentation. Turns out SKATER is pretty good out of the box - all configurations produced 0 extra fragments.

## data sources

- **NYC PLUTO** - Property land use data (building area, year built, assessed value, land use codes)
- **MTA** - Subway station locations
- **NYPD** - Crime complaint data
- **NYC 311** - Service request complaints
- **Inside Airbnb** - Listing locations and types
- **NYC DOE** - School locations and SAT scores

## stack

**Analysis**: Python, pandas, geopandas, h3-py, PySAL/spopt (SKATER), scikit-learn

**Frontend**: Next.js, React, MapLibre GL, Tailwind CSS

**Hosting**: Vercel

## what's next

The interesting extension is time-series: run this clustering year-over-year and watch neighborhoods drift. Where is gentrification actually happening according to the data? Which areas are converging vs diverging? The PLUTO archive goes back to 2002.

