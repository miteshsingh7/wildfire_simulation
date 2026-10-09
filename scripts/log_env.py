"""Print and log runtime environment and pinned dependency versions."""

from wf052.utils import get_environment_versions


def main() -> None:
    versions = get_environment_versions()
    print("========================================")
    print("wf052 Runtime Environment & Versions")
    print("========================================")
    for pkg, ver in sorted(versions.items()):
        print(f"{pkg:15s}: {ver}")
    print("========================================")


if __name__ == "__main__":
    main()
