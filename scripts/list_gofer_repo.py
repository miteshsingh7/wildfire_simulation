"""Query gee/ directory in GOFER repo."""


import requests


def main() -> None:
    url = "https://api.github.com/repos/tianjialiu/GOFER/contents/gee"
    r = requests.get(url, timeout=10)
    print("Status:", r.status_code)
    if r.status_code == 200:
        items = r.json()
        print("Files in gee/:")
        for item in items:
            print(f"  {item.get('name')} ({item.get('type')}) - {item.get('download_url')}")
            if "largeFires_metadata" in item.get("name"):
                resp = requests.get(item.get("download_url"), timeout=15)
                with open("reports/gofer_large_fires_raw.js", "w") as f:
                    f.write(resp.text)
                print(f"Downloaded {item.get('name')} ({len(resp.text)} bytes) to reports/gofer_large_fires_raw.js")

if __name__ == "__main__":
    main()
