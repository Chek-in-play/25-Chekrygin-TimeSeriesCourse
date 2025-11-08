import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:

        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """

        norm_str = ""
        if (self.is_normalize):
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"


    def _choose_distance(self):
        if self.metric.lower() == "euclidean":
            return norm_ED_distance if self.is_normalize else ED_distance
        elif self.metric.lower() == "dtw":
            return DTW_distance
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        input_data = np.array(input_data, dtype=float)
        n = input_data.shape[0]
        matrix_values = np.zeros((n, n))
        dist_func = self._choose_distance()

        for i in range(n):
            xi = z_normalize(input_data[i]) if self.is_normalize and self.metric.lower() == "dtw" else input_data[i]
            for j in range(i, n):
                xj = z_normalize(input_data[j]) if self.is_normalize and self.metric.lower() == "dtw" else input_data[j]
                dist = dist_func(xi, xj)
                matrix_values[i, j] = dist
                matrix_values[j, i] = dist

        return matrix_values

