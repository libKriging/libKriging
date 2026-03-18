# import json
import requests
import sys
import argparse


def get_package_versions(package_name):
    url = "https://pypi.org/pypi/%s/json" % (package_name,)
    try:
        with requests.get(url, allow_redirects=True) as r:
            jdata = r.json()
            # print(json.dumps(jdata, indent=2))
            versions = jdata["releases"].keys()
            return versions
    except Exception as e:
        sys.exit(e)


parser = argparse.ArgumentParser(description='Check package version')
parser.add_argument("package", metavar="PACKAGE", type=str, action='store', help="Package to check")
parser.add_argument("version", metavar="VERSION", type=str, action='store', help="Version to check")
args = parser.parse_args()

versions = get_package_versions(args.package)
if args.version in versions:
    print("Release %s of %s is available" % (args.version, args.package))
else:
    sys.exit(
        "Release %s of %s not found\nAvailable versions are: %s" % (args.version, args.package, " ".join(versions)))
