import numpy as np
from typing import Self
from modules.metrics import *
from modules.utils import z_normalize

default_metrics_params = {
    'euclidean': {'normalize': True},
    'dtw': {'normalize': True, 'r': 0.05}
}

class TimeSeriesKNN:
    """
    KNN Time Series Classifier
    """

    def __init__(self, n_neighbors: int = 3, metric: str = 'euclidean', metric_params: dict | None = None) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = default_metrics_params[metric].copy()
        if metric_params is not None:
            self.metric_params.update(metric_params)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> Self:
        """Fit the model using training data and labels"""
        self.X_train_ = X_train
        self.y_train_ = Y_train
        return self
    

        
    def _distance(self, x_train: np.ndarray, x_test: np.ndarray) -> float:
        """
        Compute distance between the train and test samples
        """
        if self.metric == "euclidean":
            if self.metric_params.get("normalize", False):
                dist = norm_ED_distance(x_train, x_test)
            else:
                dist = ED_distance(x_train, x_test)

        elif self.metric == "dtw":
            dist = DTW_distance(
                x_train, 
                x_test, 
                r=self.metric_params.get("r", 0.05)
            )

            if self.metric_params.get("normalize", False):
                from modules.utils import z_normalize
                x_train = z_normalize(x_train)
                x_test = z_normalize(x_test)
                dist = DTW_distance(
                    x_train, 
                    x_test, 
                    r=self.metric_params.get("r", 0.05)
                )
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        return dist


    def _find_neighbors(self, x_test: np.ndarray) -> list[tuple[float, int]]:
        """Find the k nearest neighbors of the test sample"""
        distances = []
        for i, x_train in enumerate(self.X_train_):
            d = self._distance(x_train, x_test)
            distances.append((d, self.y_train_[i]))
        distances.sort(key=lambda x: x[0])
        return distances[:self.n_neighbors]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for test samples"""
        y_pred = []
        for x_test in X_test:
            neighbors = self._find_neighbors(x_test)
            labels = [label for _, label in neighbors]
            most_common = max(set(labels), key=labels.count)
            y_pred.append(most_common)
        return np.array(y_pred)



def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    score = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            score += 1
    return score / len(y_true)
