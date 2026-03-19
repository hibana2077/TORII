import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


try:
    import timm
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install requirements from exp/requirements.txt first."
    ) from exc


@dataclass
class GraphData:
    image_path: Path
    class_id: int
    tokens: torch.Tensor
    patch_grid_hw: Tuple[int, int]
    assignments: np.ndarray
    super_features: torch.Tensor
    super_coords: torch.Tensor
    adjacency: torch.Tensor


def build_model(model_name: str, pretrained: bool, device: torch.device):
    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()
    model.to(device)
    return model


def preprocess_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def extract_patch_tokens(model, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    with torch.no_grad():
        feats = model.forward_features(x)

    if feats.ndim != 3:
        raise ValueError(f"Unexpected feature shape from timm model: {tuple(feats.shape)}")

    b, _, _ = feats.shape
    if b != 1:
        raise ValueError("Only batch size = 1 is supported in this experiment")

    num_prefix = getattr(model, "num_prefix_tokens", 1)
    tokens = feats[:, num_prefix:, :]

    side = int(math.sqrt(tokens.shape[1]))
    if side * side != tokens.shape[1]:
        raise ValueError(
            "Patch token count is not a square. Use a ViT-style model with square patch grid."
        )
    return tokens.squeeze(0), (side, side)


def kmeans_torch(x: torch.Tensor, n_clusters: int, n_iters: int = 30, seed: int = 0):
    n, _ = x.shape
    if n_clusters <= 0:
        raise ValueError("n_clusters must be > 0")

    n_clusters = min(n_clusters, n)
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen, device=x.device)
    centroids = x[perm[:n_clusters]].clone()

    for _ in range(n_iters):
        distances = torch.cdist(x, centroids, p=2)
        assign = torch.argmin(distances, dim=1)

        new_centroids = []
        for k in range(n_clusters):
            mask = assign == k
            if mask.any():
                new_centroids.append(x[mask].mean(dim=0))
            else:
                idx = torch.randint(0, n, (1,), generator=gen, device=x.device)
                new_centroids.append(x[idx].squeeze(0))
        new_centroids = torch.stack(new_centroids, dim=0)

        if torch.allclose(new_centroids, centroids, atol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids

    distances = torch.cdist(x, centroids, p=2)
    assign = torch.argmin(distances, dim=1)
    return assign, centroids


def build_patch_coordinates(
    grid_hw: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    grid_h, grid_w = grid_hw
    ys = torch.arange(grid_h, device=device, dtype=dtype)
    xs = torch.arange(grid_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    if grid_h > 1:
        yy = yy / float(grid_h - 1)
    else:
        yy = torch.zeros_like(yy)

    if grid_w > 1:
        xx = xx / float(grid_w - 1)
    else:
        xx = torch.zeros_like(xx)

    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)


def compute_supernode_coordinates(
    assignments: torch.Tensor,
    patch_coords: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    coords = []
    for cluster_idx in range(n_clusters):
        mask = assignments == cluster_idx
        if mask.any():
            coords.append(patch_coords[mask].mean(dim=0))
        else:
            coords.append(torch.zeros(2, device=patch_coords.device, dtype=patch_coords.dtype))
    return torch.stack(coords, dim=0)


def build_semantic_adjacency(super_features: torch.Tensor) -> torch.Tensor:
    zn = F.normalize(super_features, dim=1)
    sim = torch.mm(zn, zn.t())
    adjacency = (sim + 1.0) / 2.0
    adjacency.fill_diagonal_(0.0)
    return adjacency


def build_spatial_adjacency(
    super_coords: torch.Tensor,
    spatial_knn: int,
    spatial_sigma: float,
) -> torch.Tensor:
    if spatial_sigma <= 0:
        raise ValueError("--spatial-sigma must be > 0")

    num_nodes = super_coords.shape[0]
    distances = torch.cdist(super_coords, super_coords, p=2)
    adjacency = torch.exp(-(distances ** 2) / (2.0 * (spatial_sigma ** 2)))
    adjacency.fill_diagonal_(0.0)

    if spatial_knn > 0 and spatial_knn < num_nodes:
        knn_idx = torch.topk(distances, k=spatial_knn + 1, largest=False).indices[:, 1:]
        mask = torch.zeros_like(adjacency, dtype=torch.bool)
        row_ids = torch.arange(num_nodes, device=super_coords.device).unsqueeze(1).expand_as(knn_idx)
        mask[row_ids, knn_idx] = True
        mask = mask | mask.t()
        adjacency = adjacency * mask.to(adjacency.dtype)

    max_val = adjacency.max()
    if max_val > 0:
        adjacency = adjacency / max_val
    adjacency.fill_diagonal_(0.0)
    return adjacency


def build_super_graph(
    tokens: torch.Tensor,
    grid_hw: Tuple[int, int],
    n_clusters: int,
    seed: int,
    edge_type: str = "semantic",
    spatial_knn: int = 0,
    spatial_sigma: float = 0.35,
    hybrid_alpha: float = 0.5,
):
    if edge_type not in {"semantic", "spatial", "hybrid"}:
        raise ValueError(f"Unsupported edge_type: {edge_type}")
    if not 0.0 <= hybrid_alpha <= 1.0:
        raise ValueError("--hybrid-alpha must be in [0, 1]")

    assign, centroids = kmeans_torch(tokens, n_clusters=n_clusters, seed=seed)
    patch_coords = build_patch_coordinates(grid_hw, device=tokens.device, dtype=tokens.dtype)
    super_coords = compute_supernode_coordinates(assign, patch_coords, centroids.shape[0])

    semantic_adjacency = build_semantic_adjacency(centroids)
    spatial_adjacency = build_spatial_adjacency(super_coords, spatial_knn, spatial_sigma)

    if edge_type == "semantic":
        adjacency = semantic_adjacency
    elif edge_type == "spatial":
        adjacency = spatial_adjacency
    else:
        adjacency = hybrid_alpha * semantic_adjacency + (1.0 - hybrid_alpha) * spatial_adjacency
        adjacency.fill_diagonal_(0.0)

    return assign, centroids, super_coords, adjacency


def similarity_matrix(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    za_n = F.normalize(za, dim=1)
    zb_n = F.normalize(zb, dim=1)
    return torch.mm(za_n, zb_n.t())


def bidirectional_transport(sim: torch.Tensor, tau: float):
    p_a_from_b = torch.softmax(sim / tau, dim=1)
    p_b_from_a = torch.softmax(sim / tau, dim=0).t()
    return p_a_from_b, p_b_from_a


def compute_alignment_losses(
    ga: GraphData,
    gb: GraphData,
    p_ab: torch.Tensor,
    p_ba: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    node_b_to_a = p_ab @ gb.super_features
    node_a_to_b = p_ba @ ga.super_features
    node_loss = torch.norm(ga.super_features - node_b_to_a, p="fro") ** 2
    node_loss = node_loss + torch.norm(gb.super_features - node_a_to_b, p="fro") ** 2

    edge_b_to_a = p_ab @ gb.adjacency @ p_ab.t()
    edge_a_to_b = p_ba @ ga.adjacency @ p_ba.t()
    edge_loss = torch.norm(ga.adjacency - edge_b_to_a, p="fro") ** 2
    edge_loss = edge_loss + torch.norm(gb.adjacency - edge_a_to_b, p="fro") ** 2

    return {
        "node_loss": node_loss,
        "edge_loss": edge_loss,
    }


def optimize_learnable_transport(
    ga: GraphData,
    gb: GraphData,
    tau_p: float,
    lambda_n: float,
    lambda_e: float,
    steps: int,
    lr: float,
) -> Dict[str, float]:
    sim = similarity_matrix(ga.super_features, gb.super_features)
    logits_ab = torch.nn.Parameter(sim / tau_p)
    logits_ba = torch.nn.Parameter(sim.t() / tau_p)
    optimizer = torch.optim.Adam([logits_ab, logits_ba], lr=lr)

    final_losses = None
    final_total = None

    for iter_idx in range(max(1, steps)):
        p_ab = torch.softmax(logits_ab, dim=1)
        p_ba = torch.softmax(logits_ba, dim=1)

        losses = compute_alignment_losses(ga, gb, p_ab, p_ba)
        base_total = (
            lambda_n * losses["node_loss"]
            + lambda_e * losses["edge_loss"]
        )
        total = base_total
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        final_losses = losses
        final_total = total

        # if (steps <= 10) or (iter_idx % max(1, steps // 10) == 0):
            # print(
            #     f"  step {iter_idx + 1}/{steps}, total={total.item():.4f}, "
            #     f"node={losses['node_loss'].item():.4f}, "
            #     f"edge={losses['edge_loss'].item():.4f}"
            # )

    assert final_losses is not None and final_total is not None
    return {
        "node_loss": float(final_losses["node_loss"].item()),
        "edge_loss": float(final_losses["edge_loss"].item()),
        "total_score": float(final_total.item()),
    }


def alignment_score(
    ga: GraphData,
    gb: GraphData,
    tau_p: float,
    lambda_n: float,
    lambda_e: float,
) -> Dict[str, float]:
    sim = similarity_matrix(ga.super_features, gb.super_features)
    p_ab, p_ba = bidirectional_transport(sim, tau=tau_p)
    losses = compute_alignment_losses(ga, gb, p_ab, p_ba)
    node_loss = losses["node_loss"]
    edge_loss = losses["edge_loss"]

    total = lambda_n * node_loss + lambda_e * edge_loss
    return {
        "node_loss": float(node_loss.item()),
        "edge_loss": float(edge_loss.item()),
        "total_score": float(total.item()),
    }


def build_graph_from_image(
    model,
    image_path: Path,
    class_id: int,
    image_size: int,
    super_nodes: int,
    seed: int,
    device: torch.device,
    edge_type: str = "semantic",
    spatial_knn: int = 0,
    spatial_sigma: float = 0.35,
    hybrid_alpha: float = 0.5,
) -> GraphData:
    x = preprocess_image(image_path, image_size).to(device)
    tokens, grid_hw = extract_patch_tokens(model, x)
    assign, super_features, super_coords, adjacency = build_super_graph(
        tokens=tokens,
        grid_hw=grid_hw,
        n_clusters=super_nodes,
        seed=seed,
        edge_type=edge_type,
        spatial_knn=spatial_knn,
        spatial_sigma=spatial_sigma,
        hybrid_alpha=hybrid_alpha,
    )

    return GraphData(
        image_path=image_path,
        class_id=class_id,
        tokens=tokens.detach().cpu(),
        patch_grid_hw=grid_hw,
        assignments=assign.detach().cpu().numpy(),
        super_features=super_features.detach().cpu(),
        super_coords=super_coords.detach().cpu(),
        adjacency=adjacency.detach().cpu(),
    )
