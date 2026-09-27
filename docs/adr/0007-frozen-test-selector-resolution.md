# ADR 0007: preserve tasks while resolving test-name serialization defects

Status: development adapter, independent double controls still required.

## Observed failure and decision

The new development partition was frozen before outcomes: Moto 5164, 5255 and
6535. First admission failed before tests could run. Keep that original
manifest, freeze hashes and all original reports. Do not replace these tasks.

5164 lacks flask-cors in the isolated environment. 6535 imports NameOID from
cryptography.hazmat._oid, absent in the installed Ubuntu 3.4.8 package. The
[pinned upstream cryptography 46.0.4 source](https://github.com/pyca/cryptography/blob/46.0.4/src/cryptography/hazmat/_oid.py)
contains that class. Provision a separate rootfs with official Ubuntu
flask-cors and checksum-verified PyPI binary wheels for cryptography 46.0.4,
cffi 2.1.1, pycparser 3.0 and typing_extensions 4.15.0. Existing templates/host training libraries remain
unchanged. DNS is temporary for trusted package installation, never guest use.
The first offline bundle missed cryptography's Python <3.11 conditional
typing_extensions dependency because the downloader ran on Windows Python 3.13.
The missing pinned wheel was added, rather than allowing online dependency
resolution inside an agent or overwriting the completed older templates.

5255 has three upstream labels not matching actual pytest node names: one
Unicode-escaping difference and two parameter IDs truncated before their
closing bracket. Dropping them or running only FAIL_TO_PASS would weaken
regression coverage. An evaluation-only adapter instead collects actual tests
from base + test_patch inside the existing sandbox, **without a gold patch**.
Every required label must map to exactly one collected node: exact match first,
then Unicode escaping, then a prefix only for an incomplete parameter ID.
Missing/ambiguous IDs fail closed. Never broaden to a parent test or infer from
test outcomes. Keep original labels, patches, identities and source hashes.

Store the mapping and collection receipt privately in a separate derived
execution manifest. Freeze that derivative before running a model and rerun
unfixed/gold controls. This is a disclosed selector-equivalent adapter, not
execution of the unmodified upstream command. It does not resolve dependency
or runner errors, guarantee correct tests, or count as model success.

## Consequences and rollback

The runtime/agent never sees test_patch, gold patch, labels or mapping metadata.
Both baseline and SFT use the same frozen derived evaluator. The task pool and
development/evaluation role do not change. Retain first blocked admission and
subsequent controls as separate versions, not overwrites or cherry-picked passes.
Use the original frozen version for rollback, accepting its recorded inability
to execute under this environment. If collection/controls remain invalid, keep
the task blocked and do not substitute another task based on model success.
