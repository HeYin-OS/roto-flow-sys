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
    def query_batch(self, i_group: int, centers: np.ndarray, radius: float, max_candidates_per_point: int = 30) -> List[np.ndarray]:
        """
        批量查询候选点，并限制每个查询点的候选点数量
        
        Args:
            i_group: 组索引
            centers: 查询中心点数组，形状为(N, 2)
            radius: 查询半径
            max_candidates_per_point: 每个查询点最多返回的候选点数量，默认50
        """
        index_groups = self.tree_groups[i_group].query_ball_point(
            x=centers,
            r=radius,
            p=2,
            eps=1e-6,
            workers=-1
        )
        point_answer_groups = []
        for i, (indices, center) in enumerate(zip(index_groups, centers)):
            if len(indices) == 0:
                # 如果没有候选点，返回原始点
                point_answer_groups.append(np.array([[center[0], center[1]]], dtype=np.float32))
            elif len(indices) > max_candidates_per_point:
                # 如果候选点太多，选择距离最近的max_candidates_per_point个
                points_all = self.point_groups[i_group][indices]
                dists = np.linalg.norm(points_all - center[None, :], axis=1)
                top_indices = np.argsort(dists)[:max_candidates_per_point]
                point_answer_groups.append(points_all[top_indices])
            else:
                points_answer = self.point_groups[i_group][indices]
                point_answer_groups.append(points_answer)
        return point_answer_groups


if __name__ == '__main__':
    pass
