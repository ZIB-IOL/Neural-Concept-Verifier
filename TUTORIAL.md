# Tutorial: Running the experiments end-to-end

This walks you through a full run on **CIFAR-100**, from raw images to a trained
Prover-Verifier Game (PVG). ImageNet and CoCoLogic10 follow the same three
steps — only `--dataset` and the data location change.

```
 (1) precompute        (2) pretrain            (3) train the PVG
 concept encodings  ->  the verifier (Arthur) ->  (Merlin + Morgana + Arthur)
   *.h5                   best_model.pth            completeness / soundness
```

The whole flow below has been run end-to-end on CIFAR-100; the commands and the
file paths they produce are the real ones.

## 0. Setup

```bash
conda activate neural-concept-verifier      # see README for env creation
# Pick two directories:
export DATA_ROOT=./data            # where raw datasets live (CIFAR-100 auto-downloads here)
export EMB_DIR=./embeddings        # where concept encodings are written AND read from
```

> **The one path rule that matters.** Step 1 *writes* embeddings under
> `EMB_DIR/embeddings_<dataset>/vocab_<N>/nonsparse/…` and steps 2–3 *read* from
> `root_dir/embeddings_<dataset>/vocab_<N>/nonsparse/…`. So **`--save_dir` (step 1)
> and `--root_dir` (steps 2–3) must point to the same directory** (`$EMB_DIR`).

## 1. Precompute concept encodings

Turn images into per-concept cosine-similarity vectors (the "nonsparse"
encodings NCV uses):

```bash
python src/data/precompute_embeddings.py \
    --dataset_name cifar100 \
    --dataset_root "$DATA_ROOT" \
    --save_dir "$EMB_DIR" \
    --vocab_size 10000 \
    --nonsparse
```

**Produces:**
```
$EMB_DIR/embeddings_cifar100/vocab_10000/nonsparse/cifar100_splice_embeddings_full.h5
```
The `.h5` holds `train_embeddings/test_embeddings` (shape `N × vocab_size`) and
`train_labels/test_labels`. On first run, SpLiCE downloads its vocabulary and
CLIP image-mean to `~/.cache/splice/`.

> For **sparse** SpLiCE embeddings instead, drop `--nonsparse` and pass
> `--use_github_splice --l1_penalties 0.1` (writes under `…/vocab_10000/l1_0.100/`).

## 2. Pretrain the verifier (Arthur)

Train the classifier on the full concept encodings (no feature selection yet):

```bash
python src/main.py \
    --approach regular --epochs 30 \
    --dataset cifar100 --batch_size 256 \
    --model nonlinear --lr 1e-3 \
    --use_nonsparse --vocab_size 10000 \
    --root_dir "$EMB_DIR" \
    --save_model
```

**Saves the checkpoint to:**
```
src/checkpoints/cifar100/vocab_10000/nonsparse/nonlinear/best_model.pth
```
(Path pattern: `src/checkpoints/<dataset>/vocab_<N>/<nonsparse|l1_X>/<model>/best_model.pth`.)

## 3. Train the Prover-Verifier Game (NCV)

Now train Merlin (cooperative prover) and Morgana (adversary) against Arthur,
**warm-starting Arthur from step 2's checkpoint**:

```bash
python src/main.py \
    --approach nn --epochs 30 \
    --dataset cifar100 --batch_size 256 \
    --model nonlinear --lr 1e-3 \
    --use_nonsparse --vocab_size 10000 \
    --root_dir "$EMB_DIR" \
    --feature_selector_architecture mlp --mask_size 32 \
    --lr_merlin 0.05 --lr_morgana 0.05 --gamma 1.0 \
    --pretrained_model --pretrained_path src/checkpoints \
    --save_model
```

> **⚠️ Important — `--pretrained_path src/checkpoints` is required here.**
> `--pretrained_model` *alone* looks for the checkpoint under a different naming
> convention (`…/sfw_splice_l1_0.100_mask_32/…`), does **not** find step 2's file,
> and **silently trains Arthur from scratch** — you'll see
> `WARNING: Pretrained model not found …` and completeness near 0%. Adding
> `--pretrained_path src/checkpoints` makes it load
> `src/checkpoints/cifar100/vocab_10000/nonsparse/nonlinear/best_model.pth`
> correctly (`Loading pretrained model from …`). Always check for that line.

**Key knobs:**
- `--gamma` — weight of the adversarial (Morgana) term. `gamma=0` → no robustness,
  soundness collapses; `gamma>0` (e.g. `1.0`) → the sound PVG equilibrium.
- `--mask_size` — number of concepts the prover may select (e.g. 32).
- `--feature_selector_architecture` — `mlp` or `settransformer`.

Each epoch prints **completeness** (Merlin convinces Arthur of the truth) and
**soundness** (Morgana fails to fool Arthur).

## 4. Optional

- **Quick smoke test:** add `--debug` to any `src/main.py` command to run on a
  reduced dataset in seconds (useful to check paths before a full run).
- **SFW provers (reference optimizer):** `--approach sfw` with `--lr_merlin`,
  `--l1_penalty_coefficient`, `--mask_size` (and `--pretrained_model
  --pretrained_path src/checkpoints`).
- **Extract selected masks:** add `--extract_masks` (SFW/NN).
- **Analyze concepts:** `--compute_precision_entropy` (avg precision /
  conditional entropy of the selected concepts).
- **Other datasets:** `--dataset imagenet` (set `$IMAGENET_ROOT` for the raw
  images) or `--dataset cocologic10`; see [DCR_SWEEP_INSTRUCTIONS.md](DCR_SWEEP_INSTRUCTIONS.md)
  for the DCR experiments.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: Embeddings file not found at …` | `--root_dir` ≠ step 1's `--save_dir`, or `--vocab_size` / `--use_nonsparse` don't match what you precomputed. |
| `WARNING: Pretrained model not found …`, completeness ~0% | Missing `--pretrained_path src/checkpoints` in step 3 (see the warning above). |
| Slow first run | SpLiCE/CLIP assets download once to `~/.cache/splice` and the HF cache. |
