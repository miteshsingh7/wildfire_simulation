"""Parse GOFER metadata description and identify the 28 California fires."""

import json
from pathlib import Path


def main() -> None:
    path = Path("reports/gofer_metadata.json")
    with open(path) as f:
        data = json.load(f)

    desc = data.get("metadata", {}).get("description", "")
    print("=== GOFER Description ===")
    print(desc[:3000])

if __name__ == "__main__":
    main()
