# AutoDL executor without namespaces

Status: experimental, admission tests required before real model commands.

## Context

The current AutoDL container denies user/network namespaces. The existing
bubblewrap executor cannot start; running model shell commands as host root is
not an acceptable fallback. Keep inference, tool execution and independent
evaluation on the same machine, while keeping weights/trajectories outside the
tool filesystem. Historical runs and repeat warnings must remain intact.

## Decision

Use upstream NsJail in its documented no-mount-namespace chroot mode. Trusted
provisioning creates a clean, checksum-verified Ubuntu Base template. Each run
gets a private copy and a stable, unique unprivileged UID/GID. Host-owned sticky
workspace roots protect read-only `.git` metadata; Git objects are dissociated
from the host cache. No capabilities, inherited descriptors or environment,
network sockets, namespace creation, process tracing or session escape are
allowed. NsJail and its child share a supervised process group. Bound wall time,
processes, open files, address space, individual file size, output and writable
disk usage. Agent, admission controls and evaluator use the same backend and
unknown backend names fail closed.

This is defense in depth, not a VM or a multi-tenant production security claim.
The host still shares a kernel. Do not expose secrets or run unrelated tenants
in the same sandbox. Provisioning is trusted; task setup/code is untrusted and
must never execute in the provisioning context. No host security policy changes.

## Verification and rollback

Require Python/bash smoke, filesystem/network/privilege/metadata negative tests,
bounded-output and descendant cleanup tests, then unfixed-fail/gold-pass controls
before local-model execution. Keep the original executor for platforms where it
works. Select the new backend explicitly; never fall back after a failed launch.
Revert selection to bubblewrap only on a namespace-capable machine. New NsJail
checkouts live at `<run>/sandbox/workspace`; existing run directories are not
moved or overwritten. Recorded experiment settings include rootfs/source/model
hashes. The temporary NsJail build macro for old headers is removable after the
host headers define `PR_SCHED_CORE_SCOPE_THREAD_GROUP`; core scheduling is off.

Sources: [NsJail](https://github.com/google/nsjail),
[Ubuntu Base](https://cdimage.ubuntu.com/ubuntu-base/releases/22.04/release/),
[Git clone --dissociate](https://git-scm.com/docs/git-clone).
