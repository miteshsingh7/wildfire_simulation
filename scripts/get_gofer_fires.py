"""Query GOFER GitHub repo for the list of 28 California wildfires."""


import requests


def main() -> None:
    # Try fetching fire list or table from GitHub repo
    urls = [
        "https://raw.githubusercontent.com/tianjialiu/GOFER/main/README.md",
        "https://raw.githubusercontent.com/tianjialiu/GOFER/main/data/fire_info.csv",
        "https://raw.githubusercontent.com/tianjialiu/GOFER/main/fire_list.csv",
        "https://raw.githubusercontent.com/tianjialiu/GOFER/main/fires.csv"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                print(f"=== Found {url} ===")
                print(r.text[:2000])
                if "csv" in url:
                    with open("reports/gofer_fire_list.csv", "w") as f:
                        f.write(r.text)
                    print("Saved to reports/gofer_fire_list.csv")
                    return
        except Exception as e:
            print(f"Error {url}: {e}")

if __name__ == "__main__":
    main()
