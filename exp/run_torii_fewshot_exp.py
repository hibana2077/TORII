import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from torii_core import alignment_score
from torii_core import build_graph_from_image
from torii_core import build_model
from torii_core import GraphData
from torii_core import optimize_learnable_transport


ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


def load_split_records(dataset_root: Path, split: str, images_subdir: str) -> List[Tuple[Path, str]]:
    csv_path = dataset_root / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {csv_path}")

    images_dir = dataset_root / images_subdir
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    records: List[Tuple[Path, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"CSV must have columns 'filename,label': {csv_path}")

        for row in reader:
            filename = row["filename"].strip()
            label = row["label"].strip()
            image_path = images_dir / filename
            if image_path.suffix.lower() not in ALLOWED_EXTS:
                continue
            if not image_path.exists():
                continue
            records.append((image_path, label))

    if not records:
        raise ValueError(f"No usable records found in {csv_path}")
    return records


def build_class_index(records: List[Tuple[Path, str]]) -> Dict[str, List[Path]]:
    by_class: Dict[str, List[Path]] = defaultdict(list)
    for image_path, label in records:
        by_class[label].append(image_path)
    return dict(by_class)


def choose_episode(
    class_to_images: Dict[str, List[Path]],
    way: int,
    shot: int,
    query: int,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, List[Path]]]:
    min_count = shot + query
    eligible_classes = [c for c, imgs in class_to_images.items() if len(imgs) >= min_count]
    if len(eligible_classes) < way:
        raise ValueError(
            f"Not enough classes for {way}-way {shot}-shot {query}-query. "
            f"Need >= {way} classes with >= {min_count} images, found {len(eligible_classes)}."
        )

    selected = rng.choice(np.array(eligible_classes), size=way, replace=False)
    episode: Dict[str, Dict[str, List[Path]]] = {}
    for cls in selected.tolist():
        imgs = class_to_images[cls]
        idxs = rng.choice(len(imgs), size=min_count, replace=False)
        support = [imgs[i] for i in idxs[:shot]]
        query_imgs = [imgs[i] for i in idxs[shot:]]
        episode[cls] = {"support": support, "query": query_imgs}
    return episode


def get_graph_cached(
    image_path: Path,
    class_label: str,
    label_to_id: Dict[str, int],
    graph_cache: Dict[Path, GraphData],
    model,
    image_size: int,
    super_nodes: int,
    seed_base: int,
    device: torch.device,
) -> GraphData:
    if image_path not in graph_cache:
        class_id = label_to_id[class_label]
        local_seed = seed_base + sum(image_path.name.encode("utf-8"))
        graph_cache[image_path] = build_graph_from_image(
            model=model,
            image_path=image_path,
            class_id=class_id,
            image_size=image_size,
            super_nodes=super_nodes,
            seed=local_seed,
            device=device,
        )
    return graph_cache[image_path]


def pair_score(args, ga, gb) -> Dict[str, float]:
    if args.transport_mode == "learned":
        return optimize_learnable_transport(
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
    return alignment_score(
        ga,
        gb,
        tau_p=args.tau_p,
        lambda_n=args.lambda_n,
        lambda_e=args.lambda_e,
        lambda_p=args.lambda_p,
    )


def run(args):
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.way <= 1 or args.shot <= 0 or args.query <= 0:
        raise ValueError("Require --way > 1, --shot > 0, --query > 0")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_split_records(dataset_root, args.split, args.images_subdir)
    class_to_images = build_class_index(records)

    label_to_id = {label: idx for idx, label in enumerate(sorted(class_to_images.keys()))}
    model = build_model(args.model_name, pretrained=args.pretrained, device=device)

    rng = np.random.default_rng(args.seed)
    graph_cache: Dict[Path, GraphData] = {}

    episode_rows = []
    sample_rows = []
    total_queries = 0
    total_correct = 0

    for epi in range(args.episodes):
        episode = choose_episode(
            class_to_images=class_to_images,
            way=args.way,
            shot=args.shot,
            query=args.query,
            rng=rng,
        )

        support_by_class: Dict[str, List[Path]] = {}
        query_items: List[Tuple[str, Path]] = []
        for cls, data in episode.items():
            support_by_class[cls] = data["support"]
            for qimg in data["query"]:
                query_items.append((cls, qimg))

        episode_correct = 0
        for gt_label, query_path in query_items:
            # Label lookup by parent mapping rather than filename convention.
            q_graph = get_graph_cached(
                image_path=query_path,
                class_label=gt_label,
                label_to_id=label_to_id,
                graph_cache=graph_cache,
                model=model,
                image_size=args.image_size,
                super_nodes=args.super_nodes,
                seed_base=args.seed,
                device=device,
            )

            class_distances = {}
            for cls, support_paths in support_by_class.items():
                scores = []
                for support_path in support_paths:
                    s_graph = get_graph_cached(
                        image_path=support_path,
                        class_label=cls,
                        label_to_id=label_to_id,
                        graph_cache=graph_cache,
                        model=model,
                        image_size=args.image_size,
                        super_nodes=args.super_nodes,
                        seed_base=args.seed,
                        device=device,
                    )
                    score = pair_score(args, q_graph, s_graph)
                    scores.append(score["total_score"])
                class_distances[cls] = float(np.mean(scores))

            pred_label = min(class_distances, key=class_distances.get)
            correct = int(pred_label == gt_label)
            episode_correct += correct
            total_correct += correct
            total_queries += 1

            sample_rows.append(
                {
                    "episode": epi,
                    "query_image": query_path.name,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "correct": correct,
                    **{f"score_{cls}": v for cls, v in sorted(class_distances.items())},
                }
            )

        episode_acc = episode_correct / max(1, len(query_items))
        episode_rows.append(
            {
                "episode": epi,
                "num_queries": len(query_items),
                "correct": episode_correct,
                "episode_acc": episode_acc,
            }
        )
        print(f"episode {epi + 1}/{args.episodes}: acc={episode_acc:.4f}")

    overall_acc = total_correct / max(1, total_queries)

    ep_csv = out_dir / "episode_metrics.csv"
    if episode_rows:
        with ep_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(episode_rows)

    sample_csv = out_dir / "query_predictions.csv"
    if sample_rows:
        with sample_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"dataset_root={dataset_root}\n")
        f.write(f"split={args.split}\n")
        f.write(f"episodes={args.episodes}\n")
        f.write(f"way={args.way}\n")
        f.write(f"shot={args.shot}\n")
        f.write(f"query={args.query}\n")
        f.write(f"transport_mode={args.transport_mode}\n")
        f.write(f"total_queries={total_queries}\n")
        f.write(f"total_correct={total_correct}\n")
        f.write(f"overall_acc={overall_acc:.6f}\n")
        if episode_rows:
            ep_accs = [row["episode_acc"] for row in episode_rows]
            f.write(f"episode_acc_mean={float(np.mean(ep_accs)):.6f}\n")
            f.write(f"episode_acc_std={float(np.std(ep_accs)):.6f}\n")
            ci95 = 1.96 * float(np.std(ep_accs)) / math.sqrt(len(ep_accs))
            f.write(f"episode_acc_ci95={ci95:.6f}\n")

    print("Few-shot experiment complete")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Episode metrics: {ep_csv.resolve()}")
    print(f"Query predictions: {sample_csv.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot image classification with TORII graph alignment on MiniImageNet-style CSV splits."
        )
    )
    parser.add_argument("--dataset-root", type=str, default="dataset/miniImageNet--ravi")
    parser.add_argument("--images-subdir", type=str, default="images")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=str, default="exp/output_fewshot")

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)

    parser.add_argument("--model-name", type=str, default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--super-nodes", type=int, default=8)

    parser.add_argument("--transport-mode", type=str, choices=["similarity", "learned"], default="similarity")
    parser.add_argument("--tau-p", type=float, default=0.07)
    parser.add_argument("--lambda-n", type=float, default=1.0)
    parser.add_argument("--lambda-e", type=float, default=1.0)
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--perm-steps", type=int, default=50)
    parser.add_argument("--perm-lr", type=float, default=0.05)
    parser.add_argument("--lambda-cycle", type=float, default=0.1)
    parser.add_argument("--lambda-entropy", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
