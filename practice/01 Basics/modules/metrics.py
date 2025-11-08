import numpy as np


def ED_distance(ts1, ts2) -> float:
    ts1 = np.array(ts1, dtype=float)
    ts2 = np.array(ts2, dtype=float)
    ed_dist = np.linalg.norm(ts1 - ts2)
    return ed_dist


def norm_ED_distance(ts1, ts2) -> float:
    ts1 = np.array(ts1, dtype=float)
    ts2 = np.array(ts2, dtype=float)

    # Z-нормализация
    ts1 = (ts1 - ts1.mean()) / ts1.std()
    ts2 = (ts2 - ts2.mean()) / ts2.std()

    norm_ed_dist = np.linalg.norm(ts1 - ts2)
    return norm_ed_dist


def DTW_distance(ts1, ts2, r: float = 1) -> float:
    ts1 = np.array(ts1, dtype=float)
    ts2 = np.array(ts2, dtype=float)

    n, m = len(ts1), len(ts2)
    w = max(int(r * max(n, m)), abs(n - m))  # ограничение окна

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        start_j = max(1, i - w)
        end_j = min(m + 1, i + w + 1)
        for j in range(start_j, end_j):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2  
            dtw[i, j] = cost + min(dtw[i - 1, j],    
                                   dtw[i, j - 1],    
                                   dtw[i - 1, j - 1]) 

    dtw_dist = dtw[n, m]
    return dtw_dist
