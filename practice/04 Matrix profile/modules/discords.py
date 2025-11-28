import numpy as np
from stumpy.core import apply_exclusion_zone


def top_k_discords(matrix_profile: dict, excl_zone: int, top_k: int = 3) -> dict:
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    mp_copy = matrix_profile['mp'].copy().astype(np.float64)
    mpi = matrix_profile['mpi'].astype(np.int64)

    for _ in range(top_k):
        discord_idx = np.argmax(mp_copy) 
        discord_dist = mp_copy[discord_idx]  
        nn_idx = int(mpi[discord_idx])  

        discords_idx.append(discord_idx)
        discords_dist.append(discord_dist)
        discords_nn_idx.append(nn_idx)

        apply_exclusion_zone(mp_copy, discord_idx, excl_zone, -np.inf)

    return {
        'indices': discords_idx,
        'distances': discords_dist,
        'nn_indices': discords_nn_idx
    }


def top_k_discords2(matrix_profile: np.ndarray, excl_zone: int, top_k: int = 3) -> dict:
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    mp_copy = matrix_profile[:, 0].copy().astype(np.float64)

    for _ in range(top_k):
        discord_idx = np.argmax(mp_copy)  
        discord_dist = mp_copy[discord_idx] 
        nn_idx = int(matrix_profile[discord_idx, 1])

        discords_idx.append(discord_idx)
        discords_dist.append(discord_dist)
        discords_nn_idx.append(nn_idx)

        apply_exclusion_zone(mp_copy, discord_idx, excl_zone, -np.inf)

    return {
        'indices': discords_idx,
        'distances': discords_dist,
        'nn_indices': discords_nn_idx
    }