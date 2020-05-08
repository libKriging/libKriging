# Minimal tools
for a debian/ubuntu Linux distribution
```
apt-get update && apt-get install -y build-essential g++ cmake git liblapack-dev libopenblas-dev r-base octave liboctave-dev
```

A fully featured linux distribution could be available using docker  
```
docker run -it --rm -v "$PWD":"/data" debian:bullseye /bin/bash
```
