#!/usr/bin/env bash
# Trusted environment repair; no benchmark code, API key or network permission.
set -euo pipefail
source_root=/root/autodl-tmp/nsjail-rootfs-cognito-20260927
target_root=/root/autodl-tmp/nsjail-rootfs-source-20260927
wheel_root=/root/autodl-tmp/migration-20260927/crypto-wheels
test -d "$source_root"
test ! -L "$source_root"
test ! -e "$target_root"
test "$(realpath "$source_root")" = /root/autodl-tmp/nsjail-rootfs-cognito-20260927
test ! -L "$wheel_root"
test "$(realpath "$wheel_root")" = /root/autodl-tmp/migration-20260927/crypto-wheels
(cd "$wheel_root" && sha256sum --strict -c SHA256SUMS)
install -d -m 0755 "$target_root"
cp -a --reflink=auto "$source_root/." "$target_root"
test "$(realpath "$target_root")" = /root/autodl-tmp/nsjail-rootfs-source-20260927
# Always remove provisioning DNS even when installation fails. A partial
# template is preserved for diagnosis and must not be used by the agent.
trap 'test ! -f "$target_root/etc/resolv.conf" || unlink "$target_root/etc/resolv.conf"' EXIT
cp -L /etc/resolv.conf "$target_root/etc/resolv.conf"
env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin DEBIAN_FRONTEND=noninteractive \
  /usr/sbin/chroot "$target_root" /bin/bash -c \
  'apt-get update && apt-get install -y --no-install-recommends python3-flask-cors python3-pip && apt-get clean'
install -d -m 0755 "$target_root/opt/cab-wheels"
cp -a "$wheel_root/." "$target_root/opt/cab-wheels"
env -i PATH=/usr/bin:/bin /usr/sbin/chroot "$target_root" /usr/bin/python3 -m pip install \
  --no-index --find-links /opt/cab-wheels --target /usr/local/lib/python3.10/dist-packages \
  cryptography==46.0.4 cffi==2.1.1 pycparser==3.0 typing_extensions==4.15.0
find "$target_root" -xdev -type f -perm /6000 -exec chmod a-s {} +
find "$target_root" -xdev -type d -perm /0022 -exec chmod go-w {} +
chmod 1777 "$target_root/tmp" "$target_root/dev/shm" "$target_root/var/tmp"
chmod 0755 "$target_root"
env -i PATH=/usr/bin:/bin /usr/sbin/chroot "$target_root" /usr/bin/python3 \
  -c 'import cryptography, flask_cors; from cryptography.hazmat._oid import NameOID; print("SOURCE_DEPENDENCIES_OK", cryptography.__version__)'
env -i PATH=/usr/bin:/bin /usr/sbin/chroot "$target_root" dpkg-query -W python3-flask-cors python3-pip
