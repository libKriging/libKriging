apt update
# When used inside a docker container, a good thing is to 
# add non-root user for working (root is an unsafe user for working)
apt install -y sudo
useradd -m user --shell /bin/bash && yes password | passwd user 
echo "user ALL=NOPASSWD: ALL" | EDITOR='tee -a' visudo

apt install -y openssh-server
# patch the sshd config
cat <<EOF | patch /etc/ssh/sshd_config
--- /etc/ssh/sshd_config	2021-08-11 20:02:09.000000000 +0200
+++ /etc/ssh/sshd_config.updated	2021-11-16 19:40:41.603431000 +0100
@@ -88,7 +88,7 @@
 #GatewayPorts no
 X11Forwarding yes
 #X11DisplayOffset 10
-#X11UseLocalhost yes
+X11UseLocalhost no
 #PermitTTY yes
 PrintMotd no
 #PrintLastLog yes
EOF