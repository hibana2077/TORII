import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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


CLASS_PATTERN = re.compile(r"class(\d+)")


@dataclass
class GraphData:
    image_path: Path
    class_id: int
    tokens: torch.Tensor
    patch_grid_hw: Tuple[int, int]
    assignments: np.ndarray
    super_features: torch.Tensor
    adjacency: torch.Tensor
    distance_matrix: torch.Tensor


def parse_class_id(path: Path) -> int:
    match = CLASS_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Cannot parse class id from filename: {path.name}")
    return int(match.group(1))


def load_image_paths(data_dir: Path) -> List[Path]:
    paths = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not paths:
        raise ValueError(f"No images found in {data_dir}")
    return paths


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

    b, n, c = feats.shape
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
    n, d = x.shape
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


def build_super_graph(tokens: torch.Tensor, n_clusters: int, seed: int):
    assign, centroids = kmeans_torch(tokens, n_clusters=n_clusters, seed=seed)
    zn = F.normalize(centroids, dim=1)
    sim = torch.mm(zn, zn.t())
    adjacency = (sim + 1.0) / 2.0
    adjacency.fill_diagonal_(0.0)
    return assign, centroids, adjacency


def floyd_warshall(cost: torch.Tensor) -> torch.Tensor:
    dist = cost.clone()
    n = dist.shape[0]
    for k in range(n):
        dist = torch.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
    return dist


def graph_soft_path_distance(adjacency: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    cost = 1.0 / (adjacency + eps)
    cost.fill_diagonal_(0.0)
    return floyd_warshall(cost)


def similarity_matrix(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    za_n = F.normalize(za, dim=1)
    zb_n = F.normalize(zb, dim=1)
    return torch.mm(za_n, zb_n.t())


def bidirectional_transport(sim: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
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

    path_b_to_a = p_ab @ gb.distance_matrix @ p_ab.t()
    path_a_to_b = p_ba @ ga.distance_matrix @ p_ba.t()
    path_loss = torch.norm(ga.distance_matrix - path_b_to_a, p="fro") ** 2
    path_loss = path_loss + torch.norm(gb.distance_matrix - path_a_to_b, p="fro") ** 2

    return {
        "node_loss": node_loss,
        "edge_loss": edge_loss,
        "path_loss": path_loss,
    }


def optimize_learnable_transport(
    ga: GraphData,
    gb: GraphData,
    tau_p: float,
    lambda_n: float,
    lambda_e: float,
    lambda_p: float,
    steps: int,
    lr: float,
    lambda_cycle: float,
    lambda_entropy: float,
) -> Dict[str, float]:
    sim = similarity_matrix(ga.super_features, gb.super_features)
    logits_ab = torch.nn.Parameter(sim / tau_p)
    logits_ba = torch.nn.Parameter(sim.t() / tau_p)
    optimizer = torch.optim.Adam([logits_ab, logits_ba], lr=lr)

    eye_a = torch.eye(ga.super_features.shape[0], dtype=ga.super_features.dtype)
    eye_b = torch.eye(gb.super_features.shape[0], dtype=gb.super_features.dtype)

    final_losses = None
    final_total = None
    final_cycle = None
    final_entropy = None

    for iter in range(max(1, steps)):
        p_ab = torch.softmax(logits_ab, dim=1)
        p_ba = torch.softmax(logits_ba, dim=1)

        losses = compute_alignment_losses(ga, gb, p_ab, p_ba)
        base_total = lambda_n * losses["node_loss"] + lambda_e * losses["edge_loss"] + lambda_p * losses["path_loss"]

        cycle_loss = torch.norm(p_ab @ p_ba - eye_a, p="fro") ** 2
        cycle_loss = cycle_loss + torch.norm(p_ba @ p_ab - eye_b, p="fro") ** 2

        # Encourage crisper transport while remaining differentiable.
        entropy_ab = -(p_ab * torch.log(p_ab + 1e-8)).sum() / p_ab.shape[0]
        entropy_ba = -(p_ba * torch.log(p_ba + 1e-8)).sum() / p_ba.shape[0]
        entropy = entropy_ab + entropy_ba

        total = base_total + lambda_cycle * cycle_loss + lambda_entropy * entropy
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        
        final_losses = losses
        final_total = total
        final_cycle = cycle_loss
        final_entropy = entropy

        if (steps <= 10) or (iter % (steps // 10) == 0):
            print(f"  step {iter+1}/{steps}, total={total.item():.4f}, node={losses['node_loss'].item():.4f}, edge={losses['edge_loss'].item():.4f}, path={losses['path_loss'].item():.4f}, cycle={cycle_loss.item():.4f}, entropy={entropy.item():.4f}")

    assert final_losses is not None and final_total is not None
    return {
        "node_loss": float(final_losses["node_loss"].item()),
        "edge_loss": float(final_losses["edge_loss"].item()),
        "path_loss": float(final_losses["path_loss"].item()),
        "cycle_loss": float(final_cycle.item()),
        "entropy": float(final_entropy.item()),
        "total_score": float(final_total.item()),
    }


def alignment_score(
    ga: GraphData,
    gb: GraphData,
    tau_p: float,
    lambda_n: float,
    lambda_e: float,
    lambda_p: float,
) -> Dict[str, float]:
    sim = similarity_matrix(ga.super_features, gb.super_features)
    p_ab, p_ba = bidirectional_transport(sim, tau=tau_p)
    losses = compute_alignment_losses(ga, gb, p_ab, p_ba)
    node_loss = losses["node_loss"]
    edge_loss = losses["edge_loss"]
    path_loss = losses["path_loss"]

    total = lambda_n * node_loss + lambda_e * edge_loss + lambda_p * path_loss
    return {
        "node_loss": float(node_loss.item()),
        "edge_loss": float(edge_loss.item()),
        "path_loss": float(path_loss.item()),
        "cycle_loss": 0.0,
        "entropy": 0.0,
        "total_score": float(total.item()),
    }


def save_cluster_overlay(
    image_path: Path,
    assignments: np.ndarray,
    grid_hw: Tuple[int, int],
    out_path: Path,
    title: str,
):
    image = Image.open(image_path).convert("RGB")
    grid = assignments.reshape(grid_hw)

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis("off")

    ax2.imshow(grid, cmap="tab20")
    ax2.set_title(title)
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_image_paths(Path(args.data_dir))
    model = build_model(args.model_name, pretrained=args.pretrained, device=device)

    graphs: List[GraphData] = []

    for idx, path in enumerate(image_paths):
        class_id = parse_class_id(path)
        x = preprocess_image(path, args.image_size).to(device)
        tokens, grid_hw = extract_patch_tokens(model, x)

        assign, super_features, adjacency = build_super_graph(
            tokens=tokens,
            n_clusters=args.super_nodes,
            seed=args.seed + idx,
        )
        distance_matrix = graph_soft_path_distance(adjacency)

        graph = GraphData(
            image_path=path,
            class_id=class_id,
            tokens=tokens.detach().cpu(),
            patch_grid_hw=grid_hw,
            assignments=assign.detach().cpu().numpy(),
            super_features=super_features.detach().cpu(),
            adjacency=adjacency.detach().cpu(),
            distance_matrix=distance_matrix.detach().cpu(),
        )
        graphs.append(graph)

        overlay_path = vis_dir / f"{path.stem}_clusters.png"
        save_cluster_overlay(
            image_path=path,
            assignments=graph.assignments,
            grid_hw=grid_hw,
            out_path=overlay_path,
            title=f"Patch Clusters (k={graph.super_features.shape[0]})",
        )

    rows = []
    same_scores = []
    diff_scores = []

    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            ga = graphs[i]
            gb = graphs[j]
            if args.transport_mode == "learned":
                print(f"Optimizing transport between {ga.image_path.name} and {gb.image_path.name}...")
                scores = optimize_learnable_transport(
                    ga,
                    gb,
                    tau_p=args.tau_p,
                    lambda_n=args.lambda_n,
                    lambda_e=args.lambda_e,
                    lambda_p=args.lambda_p,
                    steps=args.perm_steps,
                    lr=args.perm_lr,
                    lambda_cycle=args.lambda_cycle,
                    lambda_entropy=args.lambda_entropy,
                )
            else:
                scores = alignment_score(
                    ga,
                    gb,
                    tau_p=args.tau_p,
                    lambda_n=args.lambda_n,
                    lambda_e=args.lambda_e,
                    lambda_p=args.lambda_p,
                )
            is_same = ga.class_id == gb.class_id
            if is_same:
                same_scores.append(scores["total_score"])
            else:
                diff_scores.append(scores["total_score"])

            rows.append(
                {
                    "img_a": ga.image_path.name,
                    "class_a": ga.class_id,
                    "img_b": gb.image_path.name,
                    "class_b": gb.class_id,
                    "pair_type": "same" if is_same else "diff",
                    "transport_mode": args.transport_mode,
                    **scores,
                }
            )

    csv_path = out_dir / "pair_alignment_scores.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_path = out_dir / "summary.txt"
    same_mean = float(np.mean(same_scores)) if same_scores else float("nan")
    diff_mean = float(np.mean(diff_scores)) if diff_scores else float("nan")
    margin = diff_mean - same_mean if not math.isnan(same_mean) and not math.isnan(diff_mean) else float("nan")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"num_images={len(graphs)}\n")
        f.write(f"num_pairs={len(rows)}\n")
        f.write(f"same_pairs={len(same_scores)}\n")
        f.write(f"diff_pairs={len(diff_scores)}\n")
        f.write(f"same_mean_total_score={same_mean:.6f}\n")
        f.write(f"diff_mean_total_score={diff_mean:.6f}\n")
        f.write(f"diff_minus_same={margin:.6f}\n")
        f.write("\n")
        f.write("Interpretation: lower score means better graph alignment under this loss.\n")

    print("Experiment complete")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Pair scores: {csv_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Visuals: {vis_dir.resolve()}")


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "TORII experiment: timm patch tokens -> clustered super-node graphs -> "
            "bidirectional graph alignment for same/different classes"
        )
    )
    parser.add_argument("--data-dir", type=str, default="exp_data")
    parser.add_argument("--output-dir", type=str, default="exp/output")
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--super-nodes", type=int, default=8)
    parser.add_argument("--transport-mode", type=str, choices=["similarity", "learned"], default="learned")
    parser.add_argument("--tau-p", type=float, default=0.07)
    parser.add_argument("--lambda-n", type=float, default=1.0)
    parser.add_argument("--lambda-e", type=float, default=1.0)
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--perm-steps", type=int, default=200)
    parser.add_argument("--perm-lr", type=float, default=0.05)
    parser.add_argument("--lambda-cycle", type=float, default=0.1)
    parser.add_argument("--lambda-entropy", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
