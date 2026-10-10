"""Inspect first 4KB of feds_western_us_2019_pixels.csv via HTTP Range request."""

import requests


def main() -> None:
    url = "https://zenodo.org/api/records/15261836/files/feds_western_us_2019_pixels.csv/content"
    r = requests.get(url, headers={"Range": "bytes=0-4096"}, timeout=15)
    print("Status:", r.status_code)
    for line in r.text.splitlines()[:10]:
        print(" ", line)


if __name__ == "__main__":
    main()
