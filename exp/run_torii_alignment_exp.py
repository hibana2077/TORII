import argparse
import csv
import math
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch

from torii_core import GraphData
from torii_core import alignment_score
from torii_core import build_graph_from_image
from torii_core import build_model
from torii_core import optimize_learnable_transport


CLASS_PATTERN = re.compile(r"class(\d+)")


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
        graph = build_graph_from_image(
            model=model,
            image_path=path,
            class_id=class_id,
            image_size=args.image_size,
            super_nodes=args.super_nodes,
            seed=args.seed + idx,
            device=device,
            edge_type=args.edge_type,
            spatial_knn=args.spatial_knn,
            spatial_sigma=args.spatial_sigma,
            hybrid_alpha=args.hybrid_alpha,
        )
        graphs.append(graph)

        overlay_path = vis_dir / f"{path.stem}_clusters.png"
        save_cluster_overlay(
            image_path=path,
            assignments=graph.assignments,
            grid_hw=graph.patch_grid_hw,
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
                    steps=args.perm_steps,
                    lr=args.perm_lr,
                )
            else:
                scores = alignment_score(
                    ga,
                    gb,
                    tau_p=args.tau_p,
                    lambda_n=args.lambda_n,
                    lambda_e=args.lambda_e,
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
                    "edge_type": args.edge_type,
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
        f.write(f"edge_type={args.edge_type}\n")
        f.write(f"spatial_knn={args.spatial_knn}\n")
        f.write(f"spatial_sigma={args.spatial_sigma}\n")
        f.write(f"hybrid_alpha={args.hybrid_alpha}\n")
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
    parser.add_argument(
        "--edge-type",
        type=str,
        choices=["semantic", "spatial", "hybrid"],
        default="semantic",
    )
    parser.add_argument(
        "--spatial-knn",
        type=int,
        default=0,
        help="If > 0, keep only k nearest spatial neighbors per super-node.",
    )
    parser.add_argument(
        "--spatial-sigma",
        type=float,
        default=0.35,
        help="Gaussian width for spatial edge affinity in normalized patch coordinates.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Weight on semantic adjacency for hybrid edges; spatial weight is 1-alpha.",
    )
    parser.add_argument("--transport-mode", type=str, choices=["similarity", "learned"], default="learned")
    parser.add_argument("--tau-p", type=float, default=0.07)
    parser.add_argument("--lambda-n", type=float, default=1.0, help="Weight for node feature alignment loss")
    parser.add_argument("--lambda-e", type=float, default=1.0, help="Weight for edge alignment loss")
    parser.add_argument("--perm-steps", type=int, default=200, help="Number of optimization steps for learnable transport")
    parser.add_argument("--perm-lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
