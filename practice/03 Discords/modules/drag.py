import numpy as np
from stumpy import core, config, stump

def _get_chunks_ranges(a, shift=None):
    if len(a) == 0:
        return np.empty((0, 2), dtype=int)
    repeats = np.full(len(a), 2, dtype=int)
    diff_is_one = np.diff(a) == 1
    repeats[1:] -= diff_is_one
    repeats[:-1] -= diff_is_one
    out = np.repeat(a, repeats).reshape(-1, 2)
    out[:, 1] += 1
    if shift is not None:
        out[:, 1] += shift
    return out

def find_candidates(T, m, M_T, S_T, r, init_cands=None, right=True, finite=False):
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    k = T.shape[0] - m + 1
    is_cands = np.ones(k, dtype=bool)
    if init_cands is not None:
        is_cands[:] = np.asarray(init_cands, dtype=bool)
    T_subseq_isfinite = np.isfinite(M_T)
    if not finite:
        T_subseq_isfinite[:] = True
    is_cands[~T_subseq_isfinite] = False
    candidate_indices = np.flatnonzero(is_cands)
    for i in np.flatnonzero(T_subseq_isfinite):
        if not np.any(is_cands):
            break
        cands_idx = np.flatnonzero(is_cands)
        if right:
            non_trivial_cands_idx = cands_idx[cands_idx < max(0, i - excl_zone)]
        else:
            non_trivial_cands_idx = cands_idx[cands_idx > i + excl_zone]
        if len(non_trivial_cands_idx) == 0:
            continue
        cand_idx_chunks = _get_chunks_ranges(non_trivial_cands_idx, shift=m - 1)
        for start, stop in cand_idx_chunks:
            chunk_data = T[start:stop]
            if len(chunk_data) >= m:
                query = np.asarray(T[i:i+m], dtype=np.float64)
                A = np.asarray(chunk_data, dtype=np.float64)
                mp = stump(A, m, query, ignore_trivial=False)
                if mp is None or len(mp) == 0:
                    continue
                D = np.asarray(mp[:, 0], dtype=np.float64)
                if D.ndim == 0:
                    D = np.atleast_1d(D)
                mask = np.flatnonzero(D < r)
                chunk_cand_indices = np.arange(start, stop - m + 1)
                if len(mask) > 0:
                    affected_indices = chunk_cand_indices[mask]
                    is_cands[affected_indices] = False
                    is_cands[i] = False
    return is_cands

def refine_candidates(T, m, M_T, S_T, is_cands):
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    k = T.shape[0] - m + 1
    P = np.full(k, -np.inf, dtype=np.float64)
    I = np.full(k, -1, dtype=np.int64)
    T_float = np.asarray(T, dtype=np.float64)
    for idx in np.flatnonzero(is_cands):
        query = np.asarray(T_float[idx:idx + m], dtype=np.float64)
        mp = stump(T_float, m, query, ignore_trivial=False)
        if mp is None or len(mp) == 0:
            continue
        D = np.asarray(mp[:, 0], dtype=np.float64).copy()
        if D.ndim == 0:
            D = np.atleast_1d(D)
        core.apply_exclusion_zone(D, idx, excl_zone, val=np.inf)
        nn_idx = np.argmin(D)
        if D[nn_idx] == np.inf:
            nn_idx = -1
        P[idx] = D[nn_idx]
        I[idx] = nn_idx
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []
    while np.any(P >= 0):
        idx = int(np.argmax(P))
        discords_idx.append(idx)
        discords_dist.append(float(P[idx]))
        discords_nn_idx.append(int(I[idx]))
        core.apply_exclusion_zone(P, idx, excl_zone, -np.inf)
    return discords_idx, discords_dist, discords_nn_idx

def DRAG(data, m, r, include=None):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("data must be a 1-D numeric array")
    if len(data) < m:
        return [], [], []
    if include is None:
        include = np.ones(len(data) - m + 1, dtype=bool)
    else:
        include = np.asarray(include[:len(data) - m + 1], dtype=bool)
    preprocess_result = core.preprocess(data, m)
    T = np.asarray(preprocess_result[0], dtype=np.float64)
    M_T = np.asarray(preprocess_result[1], dtype=np.float64)
    S_T = np.asarray(preprocess_result[2], dtype=np.float64)
    is_cands = find_candidates(T, m, M_T, S_T, r, init_cands=include, right=True)
    is_cands = find_candidates(T, m, M_T, S_T, r, init_cands=is_cands, right=False)
    return refine_candidates(T, m, M_T, S_T, is_cands)
