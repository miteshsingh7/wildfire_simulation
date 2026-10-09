# Data Card: 2018 Camp Fire Feasibility Spike (wf052 Phase 0)

## 1. Domain Specification
- **Canonical Projected CRS**: `EPSG:32610` (UTM Zone 10N)
- **Bounding Box (UTM)**:
  - Easting (X): [607230.0, 640950.0] meters (Width = 33,720 m)
  - Northing (Y): [4389660.0, 4423500.0] meters (Height = 33,840 m)
- **Resolution**: 30.0 m
- **Grid Dimensions**: 1,128 rows x 1,124 columns (1,267,872 cells)
- **Unprojected Bounding Box (WGS84)**:
  - Longitude: [-121.750163, -121.349998]
  - Latitude: [39.650017, 39.949900]

---

## 2. Ingested Data Layers Summary

### Layer A: Fuel Models & Canopy Structure
- **Product & Vintage**: LANDFIRE 2016 Remap (LF 2.0.0, pre-fire fuels)
- **Source**: USGS / LANDFIRE ArcGIS ImageServer (`https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2016/...`)
- **Layers**:
  - `fbfm40`: 40 Scott & Burgan Fire Behavior Fuel Models (categorical integer raster)
  - `canopy_cover`: Percent Tree Canopy Cover (%)
  - `canopy_height`: Canopy Height (meters x 10)
  - `canopy_base_height`: Canopy Base Height (meters x 10)
  - `canopy_bulk_density`: Canopy Bulk Density (kg/m3 x 100)
- **Vintage Justification**: Strictly predates the November 8, 2018 Camp Fire ignition, preventing data leakage from post-fire burn scar mapping present in LF 2023/2024.
- **License**: Public Domain (US Government work).

### Layer B: Topography
- **Product & Vintage**: LANDFIRE Topography 2020 / USGS 3D Elevation Program (3DEP)
- **Source**: USGS ImageServer (`https://lfps.usgs.gov/arcgis/rest/services/Landfire_Topo/...`)
- **Layers**:
  - `elevation`: Elevation in meters above sea level
  - `slope`: Slope in degrees
  - `aspect`: Aspect in degrees from North
- **License**: Public Domain (US Government work).

### Layer C: Atmospheric Weather
- **Product**: High-Resolution Rapid Refresh (HRRR) v3 Surface Analysis (3 km grid)
- **Source**: NOAA / NCEP archive via AWS S3 Open Data (`noaa-hrrr-bdp-pds.s3.amazonaws.com`) accessed via `herbie-data`
- **Time Window**: 2018-11-08 14:00 UTC (06:00 PST) to 2018-11-09 02:00 UTC (18:00 PST) [12 hours]
- **Variables**: 10m U-wind, 10m V-wind, surface gust, 2m temperature, relative humidity
- **Ground Validation**:
  - Validated against **Jarbo Gap RAWS station (`JBGC1`)** from IEM HADS archive.
  - Observed sustained winds: 25-31 mph (11-14 m/s). Peak gusts: 43-50 mph (19-22 m/s). Wind direction: 35-44° (northeast downslope Diablo wind).

### Layer D: Fire Progression Observations
- **Product**: NIST Technical Note 2135 Appendix F ("Fire Spread Data", Maranghides et al., 2021)
- **Source**: National Institute of Standards and Technology (`https://doi.org/10.6028/NIST.TN.2135`)
- **Volume**: 2,818 timestamped, geolocated observation points across Butte County.
- **Ignition Event**: Earliest observation at Pulga (Feather River Canyon / Hwy 70, Lat 39.814610, Lon -121.434004) at 06:25 PST / 14:25 UTC.
- **Format**: Vector points persisted as GeoParquet (`data/tiles/observations_nist.parquet`) and rasterized minimum arrival times (`data/tiles/arrival_time_ground_truth.tif`).

### Layer E: Satellite Fire Perimeters
- **Products Attempted**:
  - **GOFER** (GOES-Observed Fire Event Representation, Zenodo 8327264): Examined and documented. The published GOFER dataset explicitly covers 28 California fires from 2019-2021 and does not include the 2018 Camp Fire.
  - **FEDS / VIIRS 375m Active Fire**: NASA FIRMS VIIRS active fire detections for 2018-11-08 used for perimeter benchmarking.

### Layer F: Road Network
- **Product**: OpenStreetMap drivable road network via OSMnx
- **Source**: OpenStreetMap contributors / Overpass API
- **License**: Open Database License (ODbL) 1.0
- **Status & Flag**: Attempted historical Overpass attic snapshot at `2018-11-01`. The historical attic server timed out / stalled under public query load; current drivable road network snapshot was downloaded and flagged in compliance with Phase 0 rules.
- **Projected Network**: 3,396 nodes, 7,583 edges saved as GeoParquet (`data/tiles/roads_nodes.parquet`, `data/tiles/roads_edges.parquet`).
