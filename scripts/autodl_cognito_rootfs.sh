#!/usr/bin/env bash
# Trusted package provisioning, never repository setup or model commands.
set -euo pipefail
source_root=/root/autodl-tmp/nsjail-rootfs-20260927
target_root=/root/autodl-tmp/nsjail-rootfs-cognito-20260927
test -d "$source_root"
test ! -L "$source_root"
test ! -e "$target_root"
test "$(realpath "$source_root")" = /root/autodl-tmp/nsjail-rootfs-20260927
install -d -m 0755 "$target_root"
cp -a --reflink=auto "$source_root/." "$target_root"
# DNS is available only during trusted APT provisioning, never guest execution.
cp -L /etc/resolv.conf "$target_root/etc/resolv.conf"
env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin DEBIAN_FRONTEND=noninteractive \
  chroot "$target_root" /bin/bash -c \
  'apt-get update && apt-get install -y --no-install-recommends python3-jose && apt-get clean'
test "$(realpath "$target_root")" = /root/autodl-tmp/nsjail-rootfs-cognito-20260927
test ! -L "$target_root/etc/resolv.conf"
unlink "$target_root/etc/resolv.conf"
find "$target_root" -xdev -type f -perm /6000 -exec chmod a-s {} +
find "$target_root" -xdev -type d -perm /0022 -exec chmod go-w {} +
chmod 1777 "$target_root/tmp" "$target_root/dev/shm" "$target_root/var/tmp"
chmod 0755 "$target_root"
env -i PATH=/usr/bin:/bin /usr/sbin/chroot "$target_root" /usr/bin/python3 \
  -c 'from jose import jws; print("JOSE_IMPORT_OK")'
env -i PATH=/usr/bin:/bin /usr/sbin/chroot "$target_root" dpkg-query -W python3-jose python3-ecdsa python3-rsa
