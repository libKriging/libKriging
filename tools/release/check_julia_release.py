"""
Check whether a package (by UUID) is already registered in the Julia
General registry, and if so, whether a given version is already taken.

Mirrors the intent of check_python_release.py (which checks pypi.org
before letting CI re-upload an existing version), adapted to General's
registry layout: a top-level Registry.toml maps package UUID -> path,
and <path>/Versions.toml lists every registered version.

Exit code 0: safe to trigger a new registration (package not yet
registered at all, or registered but this version isn't taken yet).
Exit code 1: this exact version is already registered (or the registry
could not be read) -- CI should stop rather than re-trigger Registrator.
"""
import argparse
import sys
import urllib.request

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

GENERAL_RAW = "https://raw.githubusercontent.com/JuliaRegistries/General/master"


def fetch_toml(path):
    url = f"{GENERAL_RAW}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return tomllib.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Check General registry status for a package/version")
    parser.add_argument("uuid", help="Package UUID, as found in Project.toml")
    parser.add_argument("version", help="Version to check, e.g. 1.1.0")
    args = parser.parse_args()

    try:
        registry = fetch_toml("Registry.toml")
    except Exception as e:
        sys.exit(f"Could not fetch General/Registry.toml: {e}")

    entry = registry.get("packages", {}).get(args.uuid)
    if entry is None:
        print(f"Package {args.uuid} is not registered in General yet: this will be a first registration.")
        return

    path = entry["path"]
    try:
        versions = fetch_toml(f"{path}/Versions.toml")
    except Exception as e:
        sys.exit(f"Package is registered at General/{path} but Versions.toml could not be read: {e}")

    if args.version in versions:
        sys.exit(f"Version {args.version} is already registered in General at {path}/Versions.toml")

    print(f"Version {args.version} not yet registered at General/{path} "
          f"(existing versions: {', '.join(sorted(versions)) or 'none'}). Ok to trigger registration.")


if __name__ == "__main__":
    main()
