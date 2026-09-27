#!/usr/bin/env bash
# Trusted provisioning only. Never execute a task or repository setup as root.
set -euo pipefail
archive=/root/autodl-tmp/ubuntu-base-download-20260927/ubuntu-base-22.04.5-base-amd64.tar.gz
rootfs=/root/autodl-tmp/nsjail-rootfs-20260927
expected=242cd8898b33ea806ef5f13b1076ed7c76f9f989d18384452f7166692438ff1a
test "$(sha256sum "$archive" | cut -d' ' -f1)" = "$expected"
test ! -e "$rootfs"
install -d -m 0755 "$rootfs"
tar -xzf "$archive" -C "$rootfs" --no-same-owner
cp -L /etc/resolv.conf "$rootfs/etc/resolv.conf"
for device in null zero random urandom; do
    cp -a "/dev/$device" "$rootfs/dev/$device"
done
chmod 1777 "$rootfs/tmp"
install -d -m 1777 "$rootfs/dev/shm"
env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin DEBIAN_FRONTEND=noninteractive \
    chroot "$rootfs" /bin/bash -c 'apt-get update && apt-get install -y --no-install-recommends bash git python3 python3-pytest python3-boto3 python3-responses python3-sure python3-freezegun python3-tz python3-xmltodict python3-yaml python3-werkzeug python3-jinja2 python3-cryptography python3-dateutil python3-requests python3-packaging python3-setuptools python3-flask python3-jsonschema ca-certificates && apt-get clean'
ln -s python3 "$rootfs/usr/bin/python"
# Network-capable provisioning is finished. Guest commands never see host DNS,
# host home directories, /proc, /sys, SSH credentials or GPU devices.
rm -- "$rootfs/etc/resolv.conf"
find "$rootfs" -xdev -type f -perm /6000 -exec chmod a-s {} +
find "$rootfs" -xdev -type d -perm /0022 -exec chmod go-w {} +
chmod 1777 "$rootfs/tmp" "$rootfs/dev/shm" "$rootfs/var/tmp"
chmod 0755 "$rootfs"
printf 'ROOTFS_READY %s\n' "$expected"
