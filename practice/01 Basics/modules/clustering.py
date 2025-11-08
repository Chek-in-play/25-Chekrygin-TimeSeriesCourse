import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from typing_extensions import Self

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class TimeSeriesHierarchicalClustering:
    """
    Hierarchical Clustering of time series

    Parameters
    ----------
    n_clusters: number of clusters
    method: linkage criterion.
            Options: {single, complete, average, weighted}
    """

    def __init__(self, n_clusters: int = 3, method: str = 'complete') -> None:
        self.n_clusters: int = n_clusters
        self.method: str = method
        self.linkage_matrix: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, distance_matrix: np.ndarray) -> Self:
        distance_matrix = np.array(distance_matrix, dtype=float)
        n = distance_matrix.shape[0]
        clusters = {i: [i] for i in range(n)}
        cluster_distances = distance_matrix.copy()
        np.fill_diagonal(cluster_distances, np.inf)
        self.linkage_matrix = []
        cluster_ids = list(range(n))
        next_cluster_id = n

        while len(clusters) > 1:
            i, j = np.unravel_index(np.argmin(cluster_distances), cluster_distances.shape)
            dist = cluster_distances[i, j]
            new_cluster = clusters[cluster_ids[i]] + clusters[cluster_ids[j]]
            self.linkage_matrix.append([cluster_ids[i], cluster_ids[j], dist, len(new_cluster)])
            clusters[next_cluster_id] = new_cluster
            del clusters[cluster_ids[i]]
            del clusters[cluster_ids[j]]
            cluster_ids = list(clusters.keys())
            new_dist_matrix = np.full((len(cluster_ids), len(cluster_ids)), np.inf)
            for a in range(len(cluster_ids)):
                for b in range(a + 1, len(cluster_ids)):
                    members_a = clusters[cluster_ids[a]]
                    members_b = clusters[cluster_ids[b]]
                    if self.method == 'complete':
                        d = np.max(distance_matrix[np.ix_(members_a, members_b)])
                    elif self.method == 'single':
                        d = np.min(distance_matrix[np.ix_(members_a, members_b)])
                    elif self.method == 'average':
                        d = np.mean(distance_matrix[np.ix_(members_a, members_b)])
                    else:
                        d = np.max(distance_matrix[np.ix_(members_a, members_b)])
                    new_dist_matrix[a, b] = d
                    new_dist_matrix[b, a] = d
            cluster_distances = new_dist_matrix
            next_cluster_id += 1

        self.linkage_matrix = np.array(self.linkage_matrix)

        from scipy.cluster.hierarchy import fcluster
        self.labels_ = fcluster(self.linkage_matrix, t=self.n_clusters, criterion='maxclust') - 1

        return self

    def fit_predict(self, distance_matrix: np.ndarray) -> np.ndarray:
        self.fit(distance_matrix)
        return self.labels_

    def _draw_timeseries_allclust(self, dx: pd.DataFrame, labels: np.ndarray, leaves: list[int], gs: gridspec.GridSpec, ts_hspace: int) -> None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        margin = 7
        max_cluster = len(leaves)
        leaves = leaves[::-1]

        for cnt in range(len(leaves)):
            plt.subplot(gs[cnt:cnt+1, max_cluster-ts_hspace:max_cluster])
            plt.axis("off")
            leafnode = leaves[cnt]
            ts = dx.iloc[leafnode].to_numpy()
            ts_len = ts.shape[0] - 1
            label = int(labels[leafnode])
            color_ts = colors[label % len(colors)]
            plt.plot(ts, color=color_ts)
            plt.text(ts_len+margin, 0, f'class = {label}')

    def plot_dendrogram(self, df: pd.DataFrame, labels: np.ndarray, ts_hspace: int = 12, title: str = 'Dendrogram') -> None:
        max_cluster = self.linkage_matrix.shape[0] + 1
        plt.figure(figsize=(12, 9))
        gs = gridspec.GridSpec(max_cluster, max_cluster)
        plt.subplot(gs[:, 0 : max_cluster - ts_hspace - 1])
        plt.xlabel("Distance")
        plt.ylabel("Cluster")
        plt.title(title, fontsize=16, weight='bold')
        color_threshold = np.max(self.linkage_matrix[:, 2]) * 0.7
        ddata = dendrogram(self.linkage_matrix, orientation="left", color_threshold=color_threshold, show_leaf_counts=True)
        self._draw_timeseries_allclust(df, labels, ddata["leaves"], gs, ts_hspace)
        plt.show()
