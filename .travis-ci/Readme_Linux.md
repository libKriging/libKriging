# Minimal tools
for a debian/ubuntu Linux distribution
```
# prevents from interactive config of tzdata
ln -snf /usr/share/zoneinfo/Europe/Paris /etc/localtime
apt-get update && apt-get install -y build-essential g++ cmake git liblapack-dev libopenblas-dev r-base octave liboctave-dev sudo python3-pip lcov valgrind
# if necessary to update tzdata config
ln -snf /usr/share/zoneinfo/$(curl https://ipapi.co/timezone) /etc/localtime
```

A fully featured linux distribution could be available using docker  
```
docker run -it --rm -v "$PWD":"/data" debian:bullseye /bin/bash
```
