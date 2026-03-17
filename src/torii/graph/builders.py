from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class GraphBuildConfig:
    graph_type: str = "feature_knn"
    k: int = 8
    distance: str = "cosine"
    mutual_topk: bool = True
    spatial_radius: int = 1
    semantic_source: str = "feature"


class RelationGraphBuilder:
    def __init__(self, config: GraphBuildConfig):
        self.config = config

    def build(
        self,
        features: torch.Tensor,
        grid_size: Tuple[int, int],
        attention: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.graph_type == "feature_knn":
            return self._build_feature_knn(features)
        if self.config.graph_type == "attention":
            if attention is None:
                raise ValueError("attention graph requires attention matrix")
            return self._build_attention_graph(attention)
        if self.config.graph_type == "hybrid":
            return self._build_hybrid(features, grid_size, attention)
        raise ValueError(f"Unsupported graph_type: {self.config.graph_type}")

    def _pairwise_feature_cost(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.distance == "cosine":
            normed = F.normalize(features, p=2, dim=1)
            sim = torch.matmul(normed, normed.T)
            cost = 1.0 - sim
        elif self.config.distance == "euclidean":
            cost = torch.cdist(features, features, p=2)
        else:
            raise ValueError(f"Unsupported distance: {self.config.distance}")

        cost = torch.clamp(cost, min=0.0)
        cost.fill_diagonal_(0.0)
        return cost

    def _build_feature_knn(self, features: torch.Tensor) -> torch.Tensor:
        cost = self._pairwise_feature_cost(features)
        return self._knn_sparse_cost_from_cost(cost, largest=False)

    def _build_attention_graph(self, attention: torch.Tensor) -> torch.Tensor:
        # Attention is strength: higher is closer, convert to edge cost.
        attn = attention.clone()
        attn = torch.clamp(attn, min=0.0)
        max_val = torch.max(attn)
        if max_val > 0:
            attn = attn / max_val
        attn.fill_diagonal_(0.0)

        cost = 1.0 - attn
        cost.fill_diagonal_(0.0)
        return self._knn_sparse_cost_from_cost(cost, largest=True, score_matrix=attn)

    def _build_hybrid(
        self,
        features: torch.Tensor,
        grid_size: Tuple[int, int],
        attention: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n = features.shape[0]
        feat_cost = self._pairwise_feature_cost(features)

        spatial_adj = self._spatial_adjacency(n, grid_size, self.config.spatial_radius, features.device)

        if self.config.semantic_source == "attention":
            if attention is None:
                raise ValueError("hybrid with attention semantic_source requires attention matrix")
            sem_cost = self._build_attention_graph(attention)
            sem_adj = torch.isfinite(sem_cost) & (~torch.eye(n, dtype=torch.bool, device=features.device))
        else:
            sem_cost = self._build_feature_knn(features)
            sem_adj = torch.isfinite(sem_cost) & (~torch.eye(n, dtype=torch.bool, device=features.device))

        adj = spatial_adj | sem_adj

        hybrid_cost = torch.full((n, n), float("inf"), device=features.device)
        # Spatial edges always get local feature-distance costs.
        hybrid_cost[spatial_adj] = feat_cost[spatial_adj]
        # Semantic edges overwrite/add with their method-specific costs.
        hybrid_cost[sem_adj] = sem_cost[sem_adj]
        hybrid_cost.fill_diagonal_(0.0)
        return self._symmetrize_cost(hybrid_cost)

    def _spatial_adjacency(
        self,
        n: int,
        grid_size: Tuple[int, int],
        radius: int,
        device: torch.device,
    ) -> torch.Tensor:
        h, w = grid_size
        if h * w != n:
            raise ValueError(f"grid size {grid_size} does not match token count {n}")

        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        for y in range(h):
            for x in range(w):
                i = y * w + x
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            j = ny * w + nx
                            adj[i, j] = True
        return adj | adj.T

    def _knn_sparse_cost_from_cost(
        self,
        cost: torch.Tensor,
        largest: bool,
        score_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = cost.shape[0]
        k = min(max(1, self.config.k), n - 1)

        work = score_matrix if score_matrix is not None else cost
        work = work.clone()

        if largest:
            work.fill_diagonal_(float("-inf"))
            _, idx = torch.topk(work, k=k, dim=1, largest=True)
        else:
            work.fill_diagonal_(float("inf"))
            _, idx = torch.topk(work, k=k, dim=1, largest=False)

        adj = torch.zeros((n, n), dtype=torch.bool, device=cost.device)
        rows = torch.arange(n, device=cost.device).unsqueeze(1).expand(-1, k)
        adj[rows, idx] = True

        if self.config.mutual_topk:
            adj = adj & adj.T
        else:
            adj = adj | adj.T

        sparse_cost = torch.full((n, n), float("inf"), device=cost.device)
        sparse_cost[adj] = cost[adj]
        sparse_cost.fill_diagonal_(0.0)
        return self._symmetrize_cost(sparse_cost)

    @staticmethod
    def _symmetrize_cost(cost: torch.Tensor) -> torch.Tensor:
        finite_ij = torch.isfinite(cost)
        finite_ji = torch.isfinite(cost.T)

        both = finite_ij & finite_ji
        either = finite_ij | finite_ji

        out = torch.full_like(cost, float("inf"))
        out[both] = 0.5 * (cost[both] + cost.T[both])

        only_ij = finite_ij & (~finite_ji)
        only_ji = finite_ji & (~finite_ij)
        out[only_ij] = cost[only_ij]
        out[only_ji] = cost.T[only_ji]

        out[~either] = float("inf")
        out.fill_diagonal_(0.0)
        return out
