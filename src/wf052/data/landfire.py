"""LANDFIRE data ingestion and alignment to canonical grid.

Retrieves LF 2016 Remap fuel models (FBFM40) and canopy layers (CC, CH, CBH, CBD)
plus topography (Elevation, Slope, Aspect) directly from official USGS LANDFIRE
ArcGIS ImageServer services. Aligns all layers to the canonical 30m UTM grid.
"""

import time
from pathlib import Path

import rasterio
import requests
import yaml
from rasterio.transform import from_bounds

LANDFIRE_SERVICES = {
    "fbfm40": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2016/LF2016_FBFM40_CONUS/ImageServer/exportImage",
    "elevation": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_Topo/LF2020_Elev_CONUS/ImageServer/exportImage",
    "slope": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_Topo/LF2020_SlpD_CONUS/ImageServer/exportImage",
    "aspect": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_Topo/LF2020_Asp_CONUS/ImageServer/exportImage",
    "canopy_cover": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2016/LF2016_CC_CONUS/ImageServer/exportImage",
    "canopy_height": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2016/LF2016_CH_CONUS/ImageServer/exportImage",
    "canopy_base_height": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2016/LF2016_CBH_CONUS/ImageServer/exportImage",
    "canopy_bulk_density": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2016/LF2016_CBD_CONUS/ImageServer/exportImage",
}


def download_landfire_layer(
    layer_name: str,
    bbox_utm: dict[str, float],
    grid: dict[str, int],
    raw_dir: Path,
    max_retries: int = 3,
    timeout: int = 90,
) -> Path:
    """Download single LANDFIRE layer GeoTIFF from USGS ImageServer."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"landfire_{layer_name}_raw.tif"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path

    service_url = LANDFIRE_SERVICES.get(layer_name)
    if not service_url:
        raise ValueError(f"Unknown LANDFIRE layer: {layer_name}")

    # Bounding box in UTM coordinates
    bbox_str = f"{bbox_utm['x_min']},{bbox_utm['y_min']},{bbox_utm['x_max']},{bbox_utm['y_max']}"
    size_str = f"{grid['cols']},{grid['rows']}"

    params = {
        "bbox": bbox_str,
        "bboxSR": "32610",
        "imageSR": "32610",
        "size": size_str,
        "format": "tiff",
        "f": "json",
    }

    href = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(service_url, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                href = data.get("href")
                if href:
                    break
        except Exception:
            time.sleep(2 * attempt)

    if not href:
        msg = f"Failed to export LANDFIRE layer {layer_name} after {max_retries} attempts."
        raise RuntimeError(msg)

    # Download GeoTIFF file from href
    r_file = requests.get(href, timeout=timeout)
    r_file.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r_file.content)

    return out_path


def align_and_save_layer(
    raw_path: Path,
    out_path: Path,
    layer_name: str,
    bbox_utm: dict[str, float],
    grid: dict[str, int],
    crs: str = "EPSG:32610",
) -> None:
    """Read raw GeoTIFF, ensure alignment with exact canonical bounds, and write tile."""
    with rasterio.open(raw_path) as src:
        data = src.read(1)

    # Validate or resize to exact grid dimensions
    rows, cols = grid["rows"], grid["cols"]
    if data.shape != (rows, cols):
        from scipy.ndimage import zoom

        zoom_y = rows / data.shape[0]
        zoom_x = cols / data.shape[1]
        order = 0 if layer_name == "fbfm40" else 1
        data = zoom(data, (zoom_y, zoom_x), order=order)

    transform = from_bounds(
        bbox_utm["x_min"], bbox_utm["y_min"], bbox_utm["x_max"], bbox_utm["y_max"], cols, rows
    )

    dtype = rasterio.int32 if layer_name == "fbfm40" else rasterio.float32
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(data.astype(dtype), 1)


def ingest_landfire_pipeline(config_path: str = "configs/phase0.yaml") -> dict[str, Path]:
    """Execute full LANDFIRE download and alignment pipeline for config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bbox_utm = cfg["domain"]["bbox_utm"]
    grid = cfg["domain"]["grid"]
    crs = cfg["domain"]["crs"]
    raw_dir = Path(cfg["data"]["raw_dir"])
    tiles_dir = Path(cfg["data"]["tiles_dir"])

    layers = cfg["data"]["landfire"]["layers"]
    output_tiles: dict[str, Path] = {}

    for layer in layers:
        raw_path = download_landfire_layer(layer, bbox_utm, grid, raw_dir)
        tile_path = tiles_dir / f"{layer}.tif"
        align_and_save_layer(raw_path, tile_path, layer, bbox_utm, grid, crs)
        output_tiles[layer] = tile_path

    return output_tiles


if __name__ == "__main__":
    tiles = ingest_landfire_pipeline()
    for k, v in tiles.items():
        print(f"Generated tile: {k} -> {v}")
