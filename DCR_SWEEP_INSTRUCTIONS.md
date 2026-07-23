# DCR Multi-Dataset Hyperparameter Sweep Instructions

This guide explains how to run the WandB hyperparameter sweep for the DCR baseline on CIFAR-100, ImageNet, and CoCoLogic10.

## Prerequisites

1. Make sure `wandb` is installed:
   ```bash
   pip install wandb
   ```

2. Login to WandB:
   ```bash
   wandb login
   ```

## Running the Sweep

### 1. Initialize the Sweep

From the repository root, run:

```bash
wandb sweep sweep_dcr_cifar100.yaml
```

This will output a sweep ID like: `your-entity/Concept-Learning/sweep_id`

### 2. Start Sweep Agents

Run one or more agents to execute the sweep:

```bash
wandb agent your-entity/Concept-Learning/sweep_id
```

You can run multiple agents in parallel across different machines/GPUs:

```bash
# Terminal 1
wandb agent your-entity/Concept-Learning/sweep_id

# Terminal 2 (different GPU)
wandb agent your-entity/Concept-Learning/sweep_id
```

## Sweep Configuration

The sweep searches over the following hyperparameters:

- **dataset**: [cifar100] (imagenet and cocologic10 can be uncommented in the YAML)
- **vocab_size**: [1000, 3000, 10000]
- **emb_dim**: [8, 16, 32]
- **lr**: [0.001, 0.0001]
- **lambda_task**: [0.5, 1.0]

Fixed parameters:
- **batch_size**: 128
- **epochs**: 100
- **root_dir**: /path/to/datasets

Total combinations (CIFAR-100 only): 1 × 3 × 3 × 2 × 2 = **36 runs**

To add other datasets, edit `sweep_dcr_cifar100.yaml` and uncomment `imagenet` or `cocologic10` in the dataset values.

**Note for CoCoLogic10**: The sweep is configured to optimize for `test_accuracy`, but CoCoLogic10 also reports `test_balanced_accuracy` due to class imbalance. If running a CoCoLogic-only sweep, change the metric name to `test_balanced_accuracy` in the YAML.

## Monitoring Results

View results at: https://wandb.ai/your-entity/Concept-Learning

The sweep optimizes for **test_accuracy** (maximize). For CoCoLogic10, **test_balanced_accuracy** is also tracked and reported.

## Single Run (No Sweep)

To run a single experiment with WandB logging:

### CIFAR-100
```bash
python src/train_dcr_cifar100.py \
  --dataset cifar100 \
  --root_dir /path/to/datasets \
  --vocab_size 10000 \
  --emb_dim 32 \
  --lr 0.001 \
  --lambda_task 1.0 \
  --batch_size 128 \
  --epochs 100 \
  --wandb
```

### ImageNet
```bash
# Images automatically use default location: /path/to/pytorch_datasets/
python src/train_dcr_cifar100.py \
  --dataset imagenet \
  --root_dir /path/to/datasets \
  --vocab_size 10000 \
  --emb_dim 32 \
  --lr 0.001 \
  --lambda_task 1.0 \
  --batch_size 64 \
  --epochs 100 \
  --wandb

# Or explicitly specify image location:
# python src/train_dcr_cifar100.py \
#   --dataset imagenet \
#   --root_dir /path/to/datasets \
#   --image_root_dir /path/to/pytorch_datasets/ \
#   --vocab_size 10000 \
#   --emb_dim 32 \
#   --lr 0.001 \
#   --lambda_task 1.0 \
#   --batch_size 64 \
#   --epochs 100 \
#   --wandb
```

### CoCoLogic10
```bash
# Images automatically use default location: /path/to/pytorch_datasets/coco/data
python src/train_dcr_cifar100.py \
  --dataset cocologic10 \
  --root_dir /path/to/datasets \
  --vocab_size 10000 \
  --emb_dim 32 \
  --lr 0.001 \
  --lambda_task 1.0 \
  --batch_size 64 \
  --epochs 100 \
  --wandb

# Or explicitly specify image location:
# python src/train_dcr_cifar100.py \
#   --dataset cocologic10 \
#   --root_dir /path/to/datasets \
#   --image_root_dir /path/to/pytorch_datasets/coco/data \
#   --vocab_size 10000 \
#   --emb_dim 32 \
#   --lr 0.001 \
#   --lambda_task 1.0 \
#   --batch_size 64 \
#   --epochs 100 \
#   --wandb

# Note: The COCOLogicDataset has built-in fallback mechanisms for finding images/annotations
```

## Important: Embeddings vs. Images Paths

The script separates embeddings and image locations:

- **`--root_dir`**: Always points to where embeddings are stored
  - Example: `/path/to/datasets`
  - Embeddings structure: `<root_dir>/embeddings_<dataset>/vocab_X/nonsparse/<dataset>_splice_embeddings_full.h5`

- **`--image_root_dir`** (optional): Override where images are loaded from
  - Not needed for CIFAR-100 (images in `concept-learning` subfolder)
  - ImageNet defaults to: `/path/to/pytorch_datasets/`
  - CoCoLogic10 defaults to: `/path/to/pytorch_datasets/coco/data`

## VS Code Debugging

Use the corresponding configuration in VS Code:
- **"CIFAR100: DCR Training"**
- **"ImageNet: DCR Training"** (images auto-load from `/path/to/pytorch_datasets/`)
- **"CoCoLogic10: DCR Training"** (images auto-load from `/path/to/pytorch_datasets/coco/data`)

WandB logging is enabled by default for CIFAR-100, can be enabled by uncommenting the `--wandb` flag for others.

