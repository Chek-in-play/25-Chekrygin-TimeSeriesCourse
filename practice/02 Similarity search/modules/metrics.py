import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    return np.sqrt(np.sum((ts1 - ts2) ** 2))


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate normalized Euclidean distance between two time series.

    Parameters
    ----------
    ts1: First time series.
    ts2: Second time series.

    Returns
    -------
    Normalized Euclidean distance.
    """
    n = len(ts1)
    mu1, sigma1 = np.mean(ts1), np.std(ts1)
    mu2, sigma2 = np.mean(ts2), np.std(ts2)

    if sigma1 == 0 or sigma2 == 0:
        return np.sqrt(2 * n) if sigma1 != sigma2 else 0

    dot_product = np.dot(ts1, ts2)
    dist_sq = 2 * n * (1 - (dot_product - n * mu1 * mu2) / (n * sigma1 * sigma2))

    return np.sqrt(max(0, dist_sq))


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    n = len(ts1)
    m = len(ts2)

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    window = max(int(r * max(n, m)), 0)

    for i in range(1, n + 1):
        if window == 0:
            j = i
            if j <= m:
                cost = (ts1[i - 1] - ts2[j - 1]) ** 2
                dtw_matrix[i, j] = cost + dtw_matrix[i - 1, j - 1]
        else:
            start_j = max(1, i - window)
            end_j = min(m + 1, i + window)

            for j in range(start_j, end_j):
                cost = (ts1[i - 1] - ts2[j - 1]) ** 2
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
                )

    return dtw_matrix[n, m]