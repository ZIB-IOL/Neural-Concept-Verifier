# Neural Concept Verifier (NCV)

Official implementation of **"Neural Concept Verifier: Scaling Prover-Verifier
Games via Concept Encodings"** (Turan et al., accepted at ICML 2026).

NCV unifies **Prover-Verifier Games (PVGs)** with **concept encodings** to obtain
classifiers that are both formally verifiable and interpretable on complex,
high-dimensional inputs. Raw inputs are turned into structured concept encodings
(via SpLiCE); a **prover** selects a small subset of these concepts, and a
**verifier** — a nonlinear predictor — makes its decision using *only* that
subset. An **adversarial prover** is trained jointly to keep the selected
evidence trustworthy (high *soundness*), while the cooperative prover keeps it
informative (high *completeness*).

> **Naming.** The code uses the PVG roles from the Merlin–Arthur formulation:
> **Arthur** = verifier (classifier), **Merlin** = cooperative prover (feature
> selector), **Morgana** = adversarial prover. Concept encodings are the
> "nonsparse" SpLiCE embeddings: per-image cosine similarities to every concept
> in the vocabulary.

## Table of contents
- [Installation](#installation)
- [Data locations](#data-locations)
- [Repository structure](#repository-structure)
- [Usage](#usage)
- [Datasets](#datasets)
- [Models & key arguments](#models--key-arguments)
- [CLEVR-Hans (NCB) experiments](#clevr-hans-ncb-experiments)
- [Citation](#citation)

## Installation

```bash
git clone https://github.com/ZIB-IOL/Neural-Concept-Verifier.git
cd Neural-Concept-Verifier

conda env create --file environment.yml -n neural-concept-verifier
conda activate neural-concept-verifier

# Installs all dependencies, including upstream SpLiCE (from git).
pip install -r requirements.txt
```

SpLiCE (Sparse Linear Concept Embeddings) is pulled in as a normal dependency
from [AI4LIFE-GROUP/SpLiCE](https://github.com/AI4LIFE-GROUP/SpLiCE). Its
vocabulary and CLIP image-mean files are downloaded on first use to
`~/.cache/splice/`. The only local customization is a thin wrapper
([`src/splice_wrapper.py`](src/splice_wrapper.py)) whose `encode_image` returns
dense per-concept cosine similarities instead of the sparse decomposition.

## Data locations

Paths are configured via CLI arguments with neutral defaults; the most common
ones can also be set through environment variables:

| Variable | Default | Used for |
|---|---|---|
| `DATA_ROOT` | `./data` | Raw dataset root (embedding precompute) |
| `EMBEDDINGS_DIR` | `./embeddings` | Where precomputed embeddings are written |
| `IMAGENET_ROOT` | `./data/imagenet` | ImageNet images (DCR / pixel training) |
| `COCO_ROOT` | `./data/coco/data` | CoCoLogic images |
| `COCO_PRIMARY_ROOT` / `COCO_FALLBACK_ROOT` | — | Optional CoCoLogic annotation fallback |

## Repository structure

```
src/
├── main.py                         # Entry point (parse args -> BaseTrainer)
├── splice_wrapper.py               # SpLiCE with dense per-concept embeddings
├── config/                         # argparser + dataclass configs
├── data/
│   ├── precompute_embeddings.py    # Image -> concept-encoding (.h5) precompute
│   └── generate_COCOLogic.py       # CoCoLogic dataset construction
├── models/
│   ├── classifier.py               # Arthur (Linear / LowRank / MLP / NonLinear)
│   └── feature_selector_models.py  # Merlin/Morgana selector networks
├── merlin_arthur_framework/
│   ├── feature_selectors.py        # Neural & SFW prover/verifier selectors
│   └── stochastic_frank_wolfe.py   # SFW optimizer
├── trainer/trainer_framework.py    # Training loops for all approaches
└── analysis/analyze_masks.py       # Precision / entropy / concept analysis
train_dcr_cifar100.py               # DCR (deep concept reasoning) experiments
```

## Usage

> **New here?** Follow **[TUTORIAL.md](TUTORIAL.md)** for a verified, copy-paste
> end-to-end run on CIFAR-100 (precompute → pretrain → PVG). The sections below
> are the reference version.

### 1. Precompute concept encodings

```bash
python src/data/precompute_embeddings.py \
    --dataset_name cifar100 \
    --dataset_root "$DATA_ROOT" \
    --save_dir "$EMBEDDINGS_DIR" \
    --vocab_size 10000 \
    --nonsparse
```

`--nonsparse` produces the dense per-concept encodings used by NCV. Omit it (and
pass `--use_github_splice --l1_penalties 0.1`) to instead produce sparse SpLiCE
embeddings. Supported datasets: `cifar100`, `imagenet`, `cocologic10`.

### 2. Pretrain the verifier (Arthur)

```bash
python src/main.py \
    --approach regular --epochs 10 \
    --dataset cifar100 --batch_size 32 \
    --model nonlinear --lr 1e-3 \
    --use_nonsparse --vocab_size 10000 \
    --root_dir "$EMBEDDINGS_DIR" --save_model
```

### 3. Train the Prover-Verifier Game (NCV)

Neural provers (the main NCV method):

```bash
python src/main.py \
    --approach nn --epochs 20 \
    --dataset cifar100 --batch_size 32 \
    --model nonlinear --lr 1e-3 \
    --use_nonsparse --vocab_size 10000 \
    --feature_selector_architecture mlp --mask_size 32 \
    --lr_merlin 0.05 --lr_morgana 0.05 --gamma 1.0 \
    --root_dir "$EMBEDDINGS_DIR" --pretrained_model --save_model
```

Stochastic Frank-Wolfe provers (near-optimal per-instance selection, used as a
reference): use `--approach sfw` with `--lr_merlin`, `--l1_penalty_coefficient`,
and `--mask_size`.

`--gamma` weights the adversarial (Morgana) term: `gamma=0` trains the verifier
without robustness (soundness collapses); `gamma>0` yields the sound
prover-verifier equilibrium reported in the paper.

### Other approaches & tools
- `--approach posthoc` — post-hoc analysis of a trained model.
- `--approach unet` — pixel-space (UNet/ResNet) PVG baselines.
- `--extract_masks` (SFW/NN) — save selected masks to disk.
- `--compute_precision_entropy` — average precision / conditional entropy of the
  selected concepts (see [`src/analysis/analyze_masks.py`](src/analysis/analyze_masks.py)).

For DCR (deep concept reasoning) experiments see
[`train_dcr_cifar100.py`](src/train_dcr_cifar100.py) and
[`DCR_SWEEP_INSTRUCTIONS.md`](DCR_SWEEP_INSTRUCTIONS.md).

W&B logging is opt-in via `--wandb` (requires `wandb`; the run only records a
`project` name, no entity).

## Datasets

| Dataset | `--dataset` | Notes |
|---|---|---|
| CIFAR-100 | `cifar100` | Auto-downloaded by torchvision |
| ImageNet-1k | `imagenet` | Provide images via `$IMAGENET_ROOT` / `--image_root_dir` |
| CoCoLogic10 | `cocologic10` | Built from COCO; see `src/data/generate_COCOLogic.py` |

## Models & key arguments

**Arthur (`--model`):** `linear`, `lowrank`, `mlp`, `nonlinear` (LayerNorm + GELU).

**Prover/verifier game (`--approach nn`/`sfw`):**
`--feature_selector_architecture` (`mlp`, `settransformer`), `--mask_size`,
`--lr_merlin`, `--lr_morgana`, `--gamma`, `--segmentation_method` (`topk`),
`--prioritize_nonzero`, `--track_mask_statistics`.

The full, authoritative list of flags lives in
[`src/config/argparser.py`](src/config/argparser.py).

## CLEVR-Hans (NCB) experiments

The paper also evaluates the framework on **NCB (Neural Concept Binder)**
encodings of **CLEVR-Hans3**. That code lives in a companion repository:

➡️ **[ZIB-IOL/Neural-Concept-Verifier-NCB](https://github.com/ZIB-IOL/Neural-Concept-Verifier-NCB)** — the NCB / CLEVR-Hans3 experiments.

## Citation

```bibtex
% Accepted at ICML 2026. Until the PMLR proceedings are published, cite the arXiv version:
@article{turan2025neural,
  title   = {Neural Concept Verifier: Scaling Prover-Verifier Games via Concept Encodings},
  author  = {Turan, Berkant and Asadulla, Suhrab and Steinmann, David and
             Kersting, Kristian and Stammer, Wolfgang and Pokutta, Sebastian},
  journal = {arXiv preprint arXiv:2507.07532},
  year    = {2025}
}
```
