import re

with open("cmake/version.cmake", "r") as file:
    data = file.read()

version_major = re.search(r"^set\(KRIGING_VERSION_MAJOR (\d+)\)$", data, re.M)
version_minor = re.search(r"^set\(KRIGING_VERSION_MINOR (\d+)\)$", data, re.M)
version_patch = re.search(r"^set\(KRIGING_VERSION_PATCH (\d+)\)$", data, re.M)

print(f"{version_major.group(1)}.{version_minor.group(1)}.{version_patch.group(1)}")
