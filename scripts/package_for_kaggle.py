#!/usr/bin/env python3
"""Package codebase and training fire rasters for Kaggle execution (Phase 2C).

Packages:
1. wf052-code:
   - src/
   - configs/
   - pyproject.toml
2. wf052-tiles-train:
   - Static rasters (elevation.tif, fbfm40.tif, canopy_cover.tif) and hrrr_weather_hourly.parquet
     for the 7 training fires ONLY:
     ['bobcat', 'caldor', 'creek', 'glass', 'kincade', 'tamarack', 'zogg']
   - configs/teacher_v1.yaml
   - STRICT ASSERTION: All held-out and prohibited fires (Camp Fire, Slater & Devil, Walker,
     W-5 Cold Springs) are strictly excluded and verified.

Prints byte sizes of each packaged folder.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

SEVEN_TRAINING_FIRES: list[str] = [
    "bobcat",
    "caldor",
    "creek",
    "glass",
    "kincade",
    "tamarack",
    "zogg",
]

PROHIBITED_FIRES: list[str] = [
    "camp_fire_2018",
    "slater_and_devil_2020",
    "walker_2019",
    "w5_cold_springs_2020",
    "camp",
    "slater",
    "walker",
    "w5",
]

REQUIRED_FIRE_FILES: list[str] = [
    "elevation.tif",
    "fbfm40.tif",
    "canopy_cover.tif",
    "hrrr_weather_hourly.parquet",
]


def compute_folder_size_bytes(folder_path: Path) -> int:
    """Recursively calculate total size in bytes of all files in folder."""
    total_bytes = 0
    for root, _, files in os.walk(folder_path):
        for f in files:
            fp = Path(root) / f
            if not fp.is_symlink():
                total_bytes += fp.stat().st_size
    return total_bytes


def format_bytes(num_bytes: int) -> str:
    """Format byte count into human-readable string (KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB ({num_bytes:,} bytes)"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB ({num_bytes:,} bytes)"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB ({num_bytes:,} bytes)"


def build_kaggle_packages(
    repo_root: Path,
    output_base: Path,
) -> tuple[Path, Path, int, int]:
    """Build wf052-code and wf052-tiles-train packaging directories.

    Returns:
        (code_dir, tiles_dir, code_bytes, tiles_bytes)
    """
    output_base.mkdir(parents=True, exist_ok=True)

    code_dir = output_base / "wf052-code"
    tiles_dir = output_base / "wf052-tiles-train"

    if code_dir.exists():
        shutil.rmtree(code_dir)
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)

    code_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # 1. Package wf052-code
    print("[1/2] Building wf052-code package...")
    src_dir = repo_root / "src"
    configs_dir = repo_root / "configs"
    pyproject_file = repo_root / "pyproject.toml"

    assert src_dir.exists(), f"Source directory {src_dir} missing!"
    assert configs_dir.exists(), f"Configs directory {configs_dir} missing!"
    assert pyproject_file.exists(), f"pyproject.toml {pyproject_file} missing!"

    # Copy src/ excluding pycache
    shutil.copytree(
        src_dir,
        code_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".DS_Store"),
    )
    # Copy configs/
    shutil.copytree(
        configs_dir,
        code_dir / "configs",
        ignore=shutil.ignore_patterns(".DS_Store"),
    )
    # Copy pyproject.toml
    shutil.copy2(pyproject_file, code_dir / "pyproject.toml")

    # 2. Package wf052-tiles-train
    print("[2/2] Building wf052-tiles-train package...")
    dest_fires_dir = tiles_dir / "data" / "tiles" / "fires"
    dest_fires_dir.mkdir(parents=True, exist_ok=True)

    source_fires_dir = repo_root / "data" / "tiles" / "fires"
    assert source_fires_dir.exists(), f"Fires directory {source_fires_dir} missing!"

    # Strictly copy ONLY the 7 training fires
    for fire in SEVEN_TRAINING_FIRES:
        # Strict validation
        clean_name = fire.lower().replace("-", "_").strip()
        for prohibited in PROHIBITED_FIRES:
            assert prohibited not in clean_name, f"Prohibited fire detected in training list: {fire}!"

        src_fire_p = source_fires_dir / fire
        assert src_fire_p.exists(), f"Training fire directory {src_fire_p} missing!"

        dest_fire_p = dest_fires_dir / fire
        dest_fire_p.mkdir(parents=True, exist_ok=True)

        for fname in REQUIRED_FIRE_FILES:
            src_f = src_fire_p / fname
            assert src_f.exists(), f"Required file {fname} missing for fire {fire} at {src_f}!"
            dest_f = dest_fire_p / fname
            shutil.copy2(src_f, dest_f)

    # Copy teacher_v1.yaml
    dest_configs = tiles_dir / "configs"
    dest_configs.mkdir(parents=True, exist_ok=True)
    teacher_cfg = repo_root / "configs" / "teacher_v1.yaml"
    assert teacher_cfg.exists(), f"teacher_v1.yaml missing at {teacher_cfg}!"
    shutil.copy2(teacher_cfg, dest_configs / "teacher_v1.yaml")

    # STRICT FINAL VERIFICATION ASSERTIONS:
    # Scan all files in tiles_dir to ensure NO prohibited domains or extra fires exist
    copied_fires = [p.name for p in dest_fires_dir.iterdir() if p.is_dir()]
    assert set(copied_fires) == set(SEVEN_TRAINING_FIRES), (
        f"Copied fires mismatch! Expected {set(SEVEN_TRAINING_FIRES)}, got {set(copied_fires)}"
    )

    for root, dirs, files in os.walk(tiles_dir):
        path_str = str(Path(root)).lower().replace("-", "_")
        for proh in ["camp_fire", "slater", "walker", "w5_cold"]:
            assert proh not in path_str, f"CRITICAL: Prohibited fire {proh} found in path: {root}"

    code_size = compute_folder_size_bytes(code_dir)
    tiles_size = compute_folder_size_bytes(tiles_dir)

    print("\n" + "=" * 60)
    print("Kaggle Packaging Summary (Phase 2C)")
    print("=" * 60)
    print(f"wf052-code path:        {code_dir}")
    print(f"wf052-code total size:  {format_bytes(code_size)}")
    print("-" * 60)
    print(f"wf052-tiles-train path: {tiles_dir}")
    print(f"wf052-tiles-train size: {format_bytes(tiles_size)}")
    print(f"Training fires included: {', '.join(SEVEN_TRAINING_FIRES)} (exactly 7)")
    print("All held-out fires (Camp Fire, Slater, Walker, W-5): STRICTLY EXCLUDED & VERIFIED")
    print("=" * 60 + "\n")

    return code_dir, tiles_dir, code_size, tiles_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Package code and training tiles for Kaggle.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/kaggle"),
        help="Target directory to create packaged folders (default: build/kaggle)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = (repo_root / args.output_dir).resolve()

    build_kaggle_packages(repo_root, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
