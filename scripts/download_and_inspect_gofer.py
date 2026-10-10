"""Download GOFER.zip from Zenodo record 14642378 and inspect the 7 training fires."""

import zipfile
from pathlib import Path

import requests


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "GOFER.zip"
    extract_dir = raw_dir / "gofer"

    url = "https://zenodo.org/api/records/14642378/files/GOFER.zip/content"

    if not zip_path.exists() or zip_path.stat().st_size < 70_000_000:
        print(f"Downloading GOFER.zip from {url}...")
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"GOFER.zip already exists ({zip_path.stat().st_size / 1e6:.2f} MB)")

    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        print(f"Extracting {zip_path} to {extract_dir}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("Extraction complete.")

    all_files = sorted(extract_dir.rglob("*"))
    print(f"Total extracted items: {len(all_files)}")
    for p in all_files[:60]:
        if p.is_file():
            print(f"  {p.relative_to(extract_dir)} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
