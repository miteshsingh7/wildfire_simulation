"""Query Zenodo record 8327264 (GOFER dataset) metadata and file list."""

import json
from pathlib import Path

import requests


def main() -> None:
    url = "https://zenodo.org/api/records/8327264"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            title = data.get("metadata", {}).get("title")
            desc = data.get("metadata", {}).get("description", "")
            files = data.get("files", [])
            print(f"Title: {title}")
            print(f"Description length: {len(desc)}")
            print(f"Total files in GOFER: {len(files)}")

            # Print file names
            file_names = [f.get("key") for f in files]
            print("\nSample files:")
            for fn in file_names[:25]:
                print(f"  {fn}")

            out_path = Path("reports/gofer_metadata.json")
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nSaved metadata to {out_path}")
        else:
            print(f"HTTP status: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
