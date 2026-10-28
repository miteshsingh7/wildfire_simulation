#!/usr/bin/env python3
"""Package codebase and training fire rasters for Kaggle execution (Phase 2E).

Packages:
1. wf052-code:
   - src/
   - configs/
   - pyproject.toml
   - GIT_COMMIT.txt (HEAD commit hash and dirty status)
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
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def get_git_commit_info(repo_root: Path, allow_dirty: bool = False) -> tuple[str, bool]:
    """Retrieve HEAD commit hash and verify if the working tree is clean.

    Raises:
        RuntimeError if repository is dirty and allow_dirty is False.
    """
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        commit_hash = res.stdout.strip()
    except Exception as exc:
        commit_hash = f"unknown_{exc}"

    try:
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        status_out = status_res.stdout.strip()
        is_dirty = bool(status_out)
    except Exception:
        is_dirty = True
        status_out = "error checking status"

    if is_dirty and not allow_dirty:
        raise RuntimeError(
            f"Refusing to package from a dirty git tree! Uncommitted changes:\n{status_out}\n"
            "Please commit all changes before packaging or specify --allow-dirty."
        )

    return commit_hash, is_dirty

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


EXPECTED_TEACHER_V2_HASH: str = (
    "a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5"
)


def compute_file_sha256(file_path: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_wf052_code_zip(
    repo_root: Path,
    zip_dest: Path,
    allow_dirty: bool = True,
) -> tuple[int, str]:
    """Package code into wf052-code.zip excluding data, caches, .venv, checkpoints.

    Returns:
        (size_bytes, sha256_hash)
    """
    commit_hash, is_dirty = get_git_commit_info(repo_root, allow_dirty=allow_dirty)

    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    if zip_dest.exists():
        zip_dest.unlink()

    included_dirs = ["src", "configs", "scripts", "app"]
    included_files = ["pyproject.toml", "README.md", "REPORT.md"]

    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write GIT_COMMIT.txt into archive root
        zf.writestr("GIT_COMMIT.txt", f"commit: {commit_hash}\ndirty: {is_dirty}\n")

        for inc_dir in included_dirs:
            base_p = repo_root / inc_dir
            if not base_p.exists():
                continue
            for root, _, files in os.walk(base_p):
                for f in sorted(files):
                    fp = Path(root) / f
                    rel_path = fp.relative_to(repo_root)
                    str_rel = str(rel_path)
                    # Filter out caches, compiled files, and hidden files
                    if (
                        "__pycache__" in str_rel
                        or str_rel.endswith(".pyc")
                        or ".pytest_cache" in str_rel
                        or ".DS_Store" in str_rel
                    ):
                        continue
                    zf.write(fp, arcname=str(rel_path))

        for inc_f in included_files:
            fp = repo_root / inc_f
            if fp.exists():
                zf.write(fp, arcname=inc_f)

    size_bytes = zip_dest.stat().st_size
    sha256_hash = compute_file_sha256(zip_dest)
    return size_bytes, sha256_hash


def build_wf052_teacher_config_dataset(repo_root: Path, out_dir: Path) -> tuple[Path, int, str]:
    """Create out/wf052-teacher-config/ containing ONLY teacher_v2_a6a92f81.yaml.

    Returns:
        (target_file, file_size, sha256_hash)
    """
    target_dir = out_dir / "wf052-teacher-config"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    src_cfg = repo_root / "configs" / "teacher_v2.yaml"
    assert src_cfg.exists(), f"teacher_v2.yaml missing at {src_cfg}!"

    dest_cfg = target_dir / "teacher_v2_a6a92f81.yaml"
    shutil.copy2(src_cfg, dest_cfg)

    file_sha256 = compute_file_sha256(dest_cfg)
    assert file_sha256 == EXPECTED_TEACHER_V2_HASH, (
        f"Teacher config hash mismatch! Expected {EXPECTED_TEACHER_V2_HASH}, got {file_sha256}"
    )

    # Strictly assert ONLY this one file exists in the directory
    dir_files = [p.name for p in target_dir.iterdir() if p.is_file()]
    assert dir_files == ["teacher_v2_a6a92f81.yaml"], (
        f"Directory must contain ONLY teacher_v2_a6a92f81.yaml! Found: {dir_files}"
    )

    return dest_cfg, dest_cfg.stat().st_size, file_sha256


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
    allow_dirty: bool = True,
) -> tuple[Path, Path, int, int]:
    """Build wf052-code and wf052-tiles-train packaging directories.

    Returns:
        (code_dir, tiles_dir, code_bytes, tiles_bytes)
    """
    commit_hash, is_dirty = get_git_commit_info(repo_root, allow_dirty=allow_dirty)

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

    app_dir = repo_root / "app"
    if app_dir.exists():
        shutil.copytree(
            app_dir,
            code_dir / "app",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".DS_Store"),
        )
    for doc in ["README.md", "REPORT.md"]:
        doc_p = repo_root / doc
        if doc_p.exists():
            shutil.copy2(doc_p, code_dir / doc)

    # Write GIT_COMMIT.txt into wf052-code
    commit_info_file = code_dir / "GIT_COMMIT.txt"
    commit_info_file.write_text(f"commit: {commit_hash}\ndirty: {is_dirty}\n", encoding="utf-8")

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
    print("Kaggle Packaging Summary (Phase 2E)")
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
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow packaging from a dirty git tree (default: False)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = (repo_root / args.output_dir).resolve()

    # 1. Build directory packages (wf052-code, wf052-tiles-train)
    build_kaggle_packages(repo_root, output_dir, allow_dirty=args.allow_dirty)

    # 2. Build wf052-code.zip
    print("\n[3/4] Building wf052-code.zip...")
    zip_dest = repo_root / "wf052-code.zip"
    zip_size, zip_sha = build_wf052_code_zip(repo_root, zip_dest, allow_dirty=args.allow_dirty)
    # Also save a copy inside output_dir
    shutil.copy2(zip_dest, output_dir / "wf052-code.zip")
    print(f"wf052-code.zip path:    {zip_dest}")
    print(f"wf052-code.zip size:    {format_bytes(zip_size)}")
    print(f"wf052-code.zip SHA-256: {zip_sha}")

    # 3. Build out/wf052-teacher-config/
    print("\n[4/4] Building out/wf052-teacher-config/...")
    out_dir = repo_root / "out"
    teacher_cfg_file, cfg_size, cfg_sha = build_wf052_teacher_config_dataset(repo_root, out_dir)
    # Also copy to output_dir
    target_cfg_build = output_dir / "wf052-teacher-config"
    if target_cfg_build.exists():
        shutil.rmtree(target_cfg_build)
    shutil.copytree(out_dir / "wf052-teacher-config", target_cfg_build)
    print(f"Teacher config path:    {teacher_cfg_file}")
    print(f"Teacher config size:    {format_bytes(cfg_size)}")
    print(f"Teacher config SHA-256: {cfg_sha}")

    # 4. Print file listings
    print("\n" + "=" * 60)
    print("STEP 5 Packaging Verification: File Listings & SHA-256")
    print("=" * 60)
    print(f"Archive: {zip_dest.name} ({format_bytes(zip_size)})")
    print(f"  SHA-256: {zip_sha}")
    with zipfile.ZipFile(zip_dest, "r") as zf:
        infolist = zf.infolist()
        print(f"  Total files in zip: {len(infolist)}")
        sample_entries = [info.filename for info in infolist[:15]]
        for se in sample_entries:
            print(f"    - {se}")
        if len(infolist) > 15:
            print(f"    ... and {len(infolist) - 15} more files.")

    print(f"\nDirectory: {out_dir / 'wf052-teacher-config'}")
    for item in sorted((out_dir / "wf052-teacher-config").iterdir()):
        sz = item.stat().st_size
        sha = compute_file_sha256(item)
        print(f"  - {item.name:<30} {format_bytes(sz):>15}  SHA-256: {sha}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
