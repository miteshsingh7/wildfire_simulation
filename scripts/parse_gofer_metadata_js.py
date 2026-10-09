"""Fetch and parse largeFires_metadata.js from GOFER repo."""

import re

import requests


def main() -> None:
    url = "https://raw.githubusercontent.com/tianjialiu/GOFER/main/largeFires_metadata.js"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        text = r.text
        print(f"Downloaded {len(text)} bytes.")
        with open("reports/gofer_large_fires_raw.js", "w") as f:
            f.write(text)
        print("Saved raw js to reports/gofer_large_fires_raw.js")

        # Extract fire objects
        # Format in JS is dictionary like '2019_Kincade': { ... }
        fire_blocks = re.findall(r"'(\d{4}_[A-Za-z0-9_]+)':\s*\{([^}]+)\}", text)
        print(f"Extracted {len(fire_blocks)} fire records.")
        for name, body in fire_blocks:
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            print(f"\nFire: {name}")
            for line in lines[:6]:
                print(f"  {line}")

if __name__ == "__main__":
    main()
