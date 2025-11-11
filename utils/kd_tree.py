from typing import List

from scipy.spatial import cKDTree
import numpy as np


class BatchKDTree:
    def __init__(self, points_jagged: List[np.ndarray]):
        self.point_groups = points_jagged
        self.tree_groups = []
        for points in self.point_groups:
            self.tree_groups.append(cKDTree(points))

    def query(self, i_group: int, center: np.ndarray, radius: float) -> np.ndarray:
        indices = self.tree_groups[i_group].query_ball_point(
            x=center,
            r=radius,
            p=2,
            eps=1e-6,
            workers=-1
        )
        return self.point_groups[i_group][indices]

    # return point lists with [query_len, neighbor_len, 2] with point xy-order
    def query_batch(self, i_group: int, centers: np.ndarray, radius: float) -> List[np.ndarray]:
        index_groups = self.tree_groups[i_group].query_ball_point(
            x=centers,
            r=radius,
            p=2,
            eps=1e-6,
            workers=-1
        )
        point_answer_groups = []
        for indices in index_groups:
            points_answer = self.point_groups[i_group][indices]
            point_answer_groups.append(points_answer)
        return point_answer_groups


if __name__ == '__main__':
    pass
