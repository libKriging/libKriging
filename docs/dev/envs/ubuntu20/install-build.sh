# To execute as root
apt update

# apt install -y curl
# ln -snf /usr/share/zoneinfo/$(curl https://ipapi.co/timezone) /etc/localtime
# DEBIAN_FRONTEND="noninteractive" apt install -y tzdata

echo 'tzdata tzdata/Areas select Europe' | debconf-set-selections
echo 'tzdata tzdata/Zones/Europe select Paris' | debconf-set-selections
DEBIAN_FRONTEND="noninteractive" apt install -y tzdata

# for ubuntu:18, the equivalent of liboctave-dev was octave-pkg-dev 
apt install -y build-essential g++ cmake git python3 python3-pip octave liboctave-dev r-base liblapack-dev gfortran
apt install -y lcov valgrind # advanced tools
apt install -y ccache ninja-build vim curl # convenient tools
