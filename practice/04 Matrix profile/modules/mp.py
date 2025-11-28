import numpy as np
import pandas as pd
import math

import stumpy
from stumpy import config

def compute_mp(ts1: np.ndarray, m: int, exclusion_zone: int = None, ts2: np.ndarray = None):
    """
    Compute the matrix profile

    Parameters
    ----------
    ts1: the first time series
    m: the subsequence length
    exclusion_zone: exclusion zone
    ts2: the second time series

    Returns
    -------
    output: the matrix profile structure
            (matrix profile, matrix profile index, subsequence length, exclusion zone, the first and second time series)
    """

    if ts2 is None:
        mp_result = stumpy.stump(ts1, m)
    else:
        mp_result = stumpy.stump(ts1, m, T_B=ts2)

    return {
        'mp': mp_result[:, 0].astype(np.float64),  
        'mpi': mp_result[:, 1].astype(np.int64),  
        'm': m,  
        'data': {'ts1': ts1, 'ts2': ts2}  
    }