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

# only required for ubuntu:18
## commands from linux-macos/install.sh
## add kitware server signature cf https://apt.kitware.com       
#apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common
#curl -s https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
#apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
#apt-get install -y cmake # requires cmake ≥3.13 for target_link_options

## Required to install matlab R2021 on ubuntu:18
#apt install -y libgtk2.0-0 libnss3 libx11-xcb1 libxcb-dri3-0 libdrm2 libgbm1 libatk-bridge2.0-0
#apt install -y ssh unzip
#cat <<EOF | patch /etc/ssh/sshd_config
#--- /etc/ssh/sshd_config	2021-08-11 20:02:09.000000000 +0200
#+++ /etc/ssh/sshd_config.updated	2021-11-16 19:40:41.603431000 +0100
#@@ -88,7 +88,7 @@
# #GatewayPorts no
# X11Forwarding yes
# #X11DisplayOffset 10
#-#X11UseLocalhost yes
#+X11UseLocalhost no
# #PermitTTY yes
# PrintMotd no
# #PrintLastLog yes
#EOF
#/etc/init.d/ssh restart
## if OK after unpacking matlib installer zip file, bin/glnxa64/MATLABWindow should run without error

# When used inside a docker container, a good thing is to 
# add non-root user for working (root is an unsafe user for working)
apt install -y sudo
useradd -m user --shell /bin/bash && yes password | passwd user 
echo "user ALL=NOPASSWD: ALL" | EDITOR='tee -a' visudo