"""Compile proposed California training fires table for Step 2."""

from pathlib import Path

import pandas as pd


def main() -> None:
    fires = [
        {
            "fire_name": "Kincade",
            "year": 2019,
            "counties": "Sonoma",
            "dominant_driver": "Wind-Driven (Diablo NE winds, gusts 70-90 mph)",
            "progression_dates": "2019-10-23 to 2019-10-28",
            "pre_fire_landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
            "hrrr_weather_availability": "Hourly 10m U/V, gust, 2m RH/T (AWS S3, 100% complete)",
            "perimeter_sources_and_res": "GOFER (hourly, ~2km), FEDS (12h, 375m VIIRS), GeoMAC IR (high-res)",
            "final_area_acres": 77758.0,
            "final_area_ha": 31468.0,
            "distance_from_camp_fire_km": 130.0,
            "overlap_with_camp_fire": "None (0%)",
        },
        {
            "fire_name": "Bobcat",
            "year": 2020,
            "counties": "Los Angeles",
            "dominant_driver": "Terrain-Driven / Plume (Steep canyons, San Gabriel Mtns)",
            "progression_dates": "2020-09-06 to 2020-09-20",
            "pre_fire_landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
            "hrrr_weather_availability": "Hourly 10m U/V, gust, 2m RH/T (AWS S3, 100% complete)",
            "perimeter_sources_and_res": "GOFER (hourly, ~2km), FEDS (12h, 375m VIIRS), GeoMAC IR",
            "final_area_acres": 115997.0,
            "final_area_ha": 46943.0,
            "distance_from_camp_fire_km": 600.0,
            "overlap_with_camp_fire": "None (0%)",
        },
        {
            "fire_name": "Glass",
            "year": 2020,
            "counties": "Napa, Sonoma",
            "dominant_driver": "Wind-Driven (Diablo offshore event, grass/timber spotting)",
            "progression_dates": "2020-09-27 to 2020-10-05",
            "pre_fire_landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
            "hrrr_weather_availability": "Hourly 10m U/V, gust, 2m RH/T (AWS S3, 100% complete)",
            "perimeter_sources_and_res": "GOFER (hourly, ~2km), FEDS (12h, 375m VIIRS), GeoMAC IR",
            "final_area_acres": 67484.0,
            "final_area_ha": 27310.0,
            "distance_from_camp_fire_km": 140.0,
            "overlap_with_camp_fire": "None (0%)",
        },
        {
            "fire_name": "Creek",
            "year": 2020,
            "counties": "Fresno, Madera",
            "dominant_driver": "Terrain & Plume-Dominated (Massive conifer mortality, pyrocumulus)",
            "progression_dates": "2020-09-04 to 2020-09-12",
            "pre_fire_landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
            "hrrr_weather_availability": "Hourly 10m U/V, gust, 2m RH/T (AWS S3, 100% complete)",
            "perimeter_sources_and_res": "GOFER (hourly, ~2km), FEDS (12h, 375m VIIRS), GeoMAC / NIROPS IR",
            "final_area_acres": 379895.0,
            "final_area_ha": 153738.0,
            "distance_from_camp_fire_km": 280.0,
            "overlap_with_camp_fire": "None (0%)",
        },
        {
            "fire_name": "Zogg",
            "year": 2020,
            "counties": "Shasta",
            "dominant_driver": "Wind-Driven (Downslope N-NE winds, rapid 12h run in oak/chaparral)",
            "progression_dates": "2020-09-27 to 2020-10-01",
            "pre_fire_landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
            "hrrr_weather_availability": "Hourly 10m U/V, gust, 2m RH/T (AWS S3, 100% complete)",
            "perimeter_sources_and_res": "GOFER (hourly, ~2km), FEDS (12h, 375m VIIRS), GeoMAC IR",
            "final_area_acres": 56338.0,
            "final_area_ha": 22799.0,
            "distance_from_camp_fire_km": 110.0,
            "overlap_with_camp_fire": "None (0%)",
        },
        {
            "fire_name": "Caldor",
            "year": 2021,
            "counties": "El Dorado, Amador",
            "dominant_driver": "Mixed Terrain & Wind (Cosumnes canyon run + SW wind push across crest)",
            "progression_dates": "2021-08-14 to 2021-09-02",
            "pre_fire_landfire_vintage": "LANDFIRE 2020 (LF 2.2.0) / LF 2016 Remap",
            "hrrr_weather_availability": "Hourly 10m U/V, gust, 2m RH/T (AWS S3, 100% complete)",
            "perimeter_sources_and_res": "GOFER (hourly, ~2km), FEDS (12h, 375m VIIRS), GeoMAC / NIROPS IR",
            "final_area_acres": 221835.0,
            "final_area_ha": 89773.0,
            "distance_from_camp_fire_km": 130.0,
            "overlap_with_camp_fire": "None (0%)",
        },
        {
            "fire_name": "Tamarack",
            "year": 2021,
            "counties": "Alpine, Douglas (NV)",
            "dominant_driver": "Wind-Driven High-Elevation (Thunderstorm outflow + SW winds, sage/timber)",
            "progression_dates": "2021-07-16 to 2021-07-25",
            "pre_fire_landfire_vintage": "LANDFIRE 2020 (LF 2.2.0) / LF 2016 Remap",
            "hrrr_weather_availability": "Hourly 10m U/V, gust, 2m RH/T (AWS S3, 100% complete)",
            "perimeter_sources_and_res": "GOFER (hourly, ~2km), FEDS (12h, 375m VIIRS), GeoMAC / NIFC IR",
            "final_area_acres": 68637.0,
            "final_area_ha": 27776.0,
            "distance_from_camp_fire_km": 180.0,
            "overlap_with_camp_fire": "None (0%)",
        },
    ]

    df = pd.DataFrame(fires)
    out_csv = Path("reports/proposed_training_fires.csv")
    df.to_csv(out_csv, index=False)
    print("=== Proposed Training Fires for Phase 1A Calibration ===")
    print(df[["fire_name", "year", "counties", "dominant_driver", "final_area_acres", "distance_from_camp_fire_km"]].to_string(index=False))
    print(f"\nSaved full table to {out_csv}")

if __name__ == "__main__":
    main()
