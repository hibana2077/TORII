# TORII Experiments

This folder now has a shared core module plus two experiment entry points:

- `exp/torii_core.py`: reusable graph construction and alignment utilities
- `exp/run_torii_alignment_exp.py`: pairwise same-vs-different alignment analysis
- `exp/run_torii_fewshot_exp.py`: episodic few-shot image classification

## 1. Pairwise Alignment Experiment

This experiment uses images in `exp_data` with class labels parsed from filenames like `*_class185.png`.

Pipeline:

1. Extract ViT patch tokens with timm.
2. Cluster patch tokens into super-nodes (default: 8).
3. Build a super-node graph by centroid cosine affinity.
4. Compute graph path geometry (all-pairs shortest path on inverse affinity).
5. Compute bidirectional transport alignment losses:
   - node alignment
   - edge consistency
   - path-structure alignment
   - optional learnable transport matrix optimization (gradient descent)
6. Compare same-class and different-class image pairs.

## Install

```bash
pip install -r exp/requirements.txt
```

## Run

```bash
python exp/run_torii_alignment_exp.py --data-dir exp_data --output-dir exp/output --pretrained
```

Learnable permutation/transport matrix mode (default):

```bash
python exp/run_torii_alignment_exp.py --data-dir exp_data --output-dir exp/output_learned --transport-mode learned --perm-steps 200 --perm-lr 0.05
```

Optional parameters:

- `--model-name vit_base_patch16_224`
- `--transport-mode learned|similarity`
- `--super-nodes 8`
- `--tau-p 0.07`
- `--lambda-n 1.0 --lambda-e 1.0 --lambda-p 1.0`
- `--perm-steps 200 --perm-lr 0.05`
- `--lambda-cycle 0.1 --lambda-entropy 0.01`
- `--cpu`

## Output

- `exp/output/pair_alignment_scores.csv`: pairwise scores and pair type (`same`/`diff`)
- CSV now also includes `transport_mode`, `cycle_loss`, and `entropy` columns
- `exp/output/summary.txt`: summary metrics
- `exp/output/visuals/*_clusters.png`: image + patch-cluster map

Interpretation: lower total score indicates better alignment under the defined losses.

## 2. Few-Shot Classification Experiment (MiniImageNet-style)

Expected dataset structure (matching your txt example):

```text
dataset/miniImageNet--ravi/
   images/
      n0193011200000001.jpg
      ...
   train.csv
   val.csv
   test.csv
```

Each CSV must include:

```text
filename,label
n0193011200000001.jpg,n01930112
```

Run 5-way 1-shot evaluation on `test.csv`:

```bash
python exp/run_torii_fewshot_exp.py \
   --dataset-root dataset/miniImageNet--ravi \
   --split test \
   --episodes 100 \
   --way 5 \
   --shot 1 \
   --query 5 \
   --pretrained
```

Run 5-way 5-shot with learned transport (slower):

```bash
python exp/run_torii_fewshot_exp.py \
   --dataset-root dataset/miniImageNet--ravi \
   --split test \
   --episodes 100 \
   --way 5 \
   --shot 5 \
   --query 5 \
   --transport-mode learned \
   --perm-steps 50 \
   --pretrained
```

Few-shot outputs:

- `exp/output_fewshot/episode_metrics.csv`: per-episode accuracy
- `exp/output_fewshot/query_predictions.csv`: per-query prediction and class scores
- `exp/output_fewshot/summary.txt`: overall accuracy and confidence interval
