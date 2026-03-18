# TORII Experiment (Patch Token -> Super-Node Graph Alignment)

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
