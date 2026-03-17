# TORII
Path-Aware Relation Graph Alignment

## Project Structure

- `src/torii/vit/extractor.py`: timm ViT feature and attention extractor
- `src/torii/graph/builders.py`: relation-graph construction (`feature_knn`, `attention`, `hybrid`)
- `src/torii/graph/soft_path.py`: soft path geometry computation
- `src/torii/experiments/min_exp.py`: reusable minimal experiment pipeline and CLI logic
- `min_exp.py`: entry point for image experiment on `exp_data`
- `src/torii/legacy/`: relocated synthetic experiments from old root scripts

## Run `min_exp`

Default behavior uses timm ViT with `pretrained=False`.

```bash
python min_exp.py --data-dir exp_data --graph-type feature_knn
```

Enable pretrained weights explicitly:

```bash
python min_exp.py --data-dir exp_data --graph-type feature_knn --pretrained
```

Three graph methods from `docs/main.md`:

```bash
python min_exp.py --data-dir exp_data --graph-type feature_knn --k 8
python min_exp.py --data-dir exp_data --graph-type attention --k 8 --attn-layers 2
python min_exp.py --data-dir exp_data --graph-type hybrid --k 8 --semantic-source feature --spatial-radius 1
```
