import argparse
import itertools
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from src.torii.graph import GraphBuildConfig, RelationGraphBuilder, compute_soft_path_matrix
from src.torii.vit import ViTExtractorConfig, ViTFeatureExtractor


@dataclass
class ImageGraphSample:
    path: str
    class_id: str
    graph: torch.Tensor
    soft_path: torch.Tensor
    n_edges: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TORII minimal ViT relation-graph experiment")
    parser.add_argument("--data-dir", type=str, default="exp_data")
    parser.add_argument("--graph-type", type=str, default="feature_knn", choices=["feature_knn", "attention", "hybrid"])
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--pretrained", action="store_true", help="Enable timm pretrained weights (default: False)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--distance", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--mutual-topk", action="store_true", default=True)
    parser.add_argument("--no-mutual-topk", action="store_false", dest="mutual_topk")
    parser.add_argument("--spatial-radius", type=int, default=1)
    parser.add_argument("--semantic-source", type=str, default="feature", choices=["feature", "attention"])
    parser.add_argument("--attn-layers", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--k-hops", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-images", type=int, default=4)
    return parser.parse_args()


def infer_class_id(filename: str) -> str:
    m = re.search(r"class(\d+)", filename)
    return m.group(1) if m else "unknown"


def list_images(data_dir: str, max_images: int) -> List[Tuple[str, str]]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    selected = files[:max_images]
    return [(os.path.join(data_dir, f), infer_class_id(f)) for f in selected]


def edge_count(cost_matrix: torch.Tensor) -> int:
    n = cost_matrix.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=cost_matrix.device)
    finite = torch.isfinite(cost_matrix) & off_diag
    # Undirected edge count.
    return int(finite.sum().item() // 2)


def relation_distance(d1: torch.Tensor, d2: torch.Tensor) -> float:
    denom = max(1, d1.numel())
    return float(torch.norm(d1 - d2, p="fro").item() / denom)


def run(args: argparse.Namespace) -> None:
    image_items = list_images(args.data_dir, args.max_images)
    if len(image_items) < 2:
        raise RuntimeError("Need at least 2 images for pairwise test")

    extractor = ViTFeatureExtractor(
        ViTExtractorConfig(
            model_name=args.model_name,
            pretrained=args.pretrained,
            image_size=args.image_size,
            num_attn_layers=args.attn_layers,
            device=args.device,
        )
    )

    builder = RelationGraphBuilder(
        GraphBuildConfig(
            graph_type=args.graph_type,
            k=args.k,
            distance=args.distance,
            mutual_topk=args.mutual_topk,
            spatial_radius=args.spatial_radius,
            semantic_source=args.semantic_source,
        )
    )

    samples: List[ImageGraphSample] = []
    for image_path, class_id in image_items:
        features, attention, grid_size = extractor.extract_from_image_path(image_path)

        if args.graph_type == "attention" and attention is None:
            raise RuntimeError("No attention was captured; cannot build attention graph")

        graph = builder.build(features=features, grid_size=grid_size, attention=attention)
        soft_path = compute_soft_path_matrix(graph, tau=args.tau, k_hops=args.k_hops)

        samples.append(
            ImageGraphSample(
                path=image_path,
                class_id=class_id,
                graph=graph,
                soft_path=soft_path,
                n_edges=edge_count(graph),
            )
        )

    print("=== TORII min_exp ===")
    print(f"graph_type={args.graph_type} | model={args.model_name} | pretrained={args.pretrained}")
    print(f"images={len(samples)} | tau={args.tau} | k_hops={args.k_hops} | k={args.k}")

    for i, s in enumerate(samples):
        print(f"[{i}] class={s.class_id} edges={s.n_edges} path={s.path}")

    same_dists = []
    diff_dists = []

    print("\nPairwise relation distance (Frobenius normalized on D_tau):")
    for (i, a), (j, b) in itertools.combinations(enumerate(samples), 2):
        dist = relation_distance(a.soft_path, b.soft_path)
        same = a.class_id == b.class_id
        tag = "same" if same else "diff"
        print(f"({i},{j}) [{tag}] class=({a.class_id},{b.class_id}) dist={dist:.6f}")
        if same:
            same_dists.append(dist)
        else:
            diff_dists.append(dist)

    print("\nSummary:")
    if same_dists:
        print(f"same-class mean={np.mean(same_dists):.6f} std={np.std(same_dists):.6f}")
    else:
        print("same-class mean=NA (no same-class pairs)")

    if diff_dists:
        print(f"diff-class mean={np.mean(diff_dists):.6f} std={np.std(diff_dists):.6f}")
    else:
        print("diff-class mean=NA (no diff-class pairs)")


if __name__ == "__main__":
    run(parse_args())
