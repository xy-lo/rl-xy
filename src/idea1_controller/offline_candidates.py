"""Pre-evaluated candidate artefacts with mandatory current admission.

A cache hit is valid only for the exact public batch, distribution and service
identity. Generalization to other workload buckets must be measured separately.
"""
from copy import deepcopy
from .workload import digest


def freeze_candidates(public, snapshot, catalog_sha, result):
    if not result['winner']['passed']:
        raise ValueError('unvalidated offline winner')
    value = dict(schema='idea1.offline-candidates/1', public_digest=digest(public),
        distribution_version=snapshot.version if snapshot else None,
        catalog_sha256=catalog_sha, result=deepcopy(result))
    value['artifact_digest'] = digest(value)
    return value


def select_precomputed(artifact, public, snapshot, catalog_sha, *, admit, fallback):
    raw = {k:v for k,v in artifact.items() if k != 'artifact_digest'}
    if digest(raw) != artifact.get('artifact_digest'):
        raise ValueError('offline artifact digest mismatch')
    if (artifact['public_digest'] != digest(public) or
        artifact['distribution_version'] != (snapshot.version if snapshot else None) or
        artifact['catalog_sha256'] != catalog_sha):
        return fallback(), 'IDENTITY_MISMATCH'
    result = deepcopy(artifact['result'])
    if not admit(result):
        return fallback(), 'CURRENT_ADMISSION_FAILED'
    return result, 'OFFLINE_CANDIDATE_REVALIDATED'
