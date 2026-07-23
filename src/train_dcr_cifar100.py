"""
Deep Concept Reasoning (DCR) on CIFAR-100, ImageNet, and CoCoLogic10

This script trains a DCR model using precomputed SpLiCE/CLIP 
similarity vectors as concept truth degrees.

Usage:
    python src/train_dcr_cifar100.py \
        --dataset cifar100 \
        --root_dir /path/to/data \
        --vocab_size 1000 \
        --lr 0.001 \
        --lambda_task 1.0 \
        --emb_dim 8 \
        --batch_size 128 \
        --epochs 100
"""

import argparse
import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np

# Import torch-explain components
from torch_explain.nn.concepts import ConceptEmbedding, ConceptReasoningLayer

# WandB for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Import for CoCoLogic dataset
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
try:
    from data.generate_COCOLogic import COCOLogicDataset, load_category_mapping
    COCOLOGIC_AVAILABLE = True
except ImportError:
    COCOLOGIC_AVAILABLE = False
    print("Warning: CoCoLogic dataset not available")

# Import for balanced accuracy (needed for CoCoLogic)
from sklearn.metrics import balanced_accuracy_score


class DatasetWithConcepts(Dataset):
    """
    Generic dataset wrapper that augments images with precomputed SpLiCE/CLIP similarity vectors.
    
    Supports: CIFAR-100, ImageNet, CoCoLogic10
    The dataset ensures alignment between images and concept truth degrees by
    using the same ordering as the precomputation step (shuffle=False).
    """
    
    def __init__(self, root_dir, dataset_name='cifar100', train=True, embeddings_path=None, transform=None):
        """
        Args:
            root_dir: Root directory for data
            dataset_name: Dataset name ('cifar100', 'imagenet', or 'cocologic10')
            train: Whether to load train or test split
            embeddings_path: Path to HDF5 file with precomputed similarities
            transform: Optional transform to apply to images
        """
        self.dataset_name = dataset_name.lower()
        
        # Load the appropriate dataset based on dataset_name
        if self.dataset_name == 'cifar100':
            # CIFAR-100 is located in concept-learning subdirectory
            cifar_path = os.path.join(root_dir, 'concept-learning')
            self.base_dataset = datasets.CIFAR100(
                root=cifar_path, 
                train=train, 
                download=False,
                transform=transform
            )
        elif self.dataset_name == 'imagenet':
            # ImageNet uses ImageFolder
            # Handle different root_dir formats for ImageNet
            # Case 1: root_dir already points to the image dataset root (e.g., /path/to/pytorch_datasets/)
            # Case 2: root_dir is the embeddings path (e.g., /path/to/datasets)
            split_dir = 'train' if train else 'val'
            
            if 'pytorch_datasets' in root_dir or root_dir.endswith('imagenet'):
                # Direct path to pytorch_datasets or imagenet folder
                if root_dir.endswith('imagenet'):
                    imagenet_path = os.path.join(root_dir, split_dir)
                else:
                    imagenet_path = os.path.join(root_dir, 'imagenet', split_dir)
            else:
                # Fallback: Try standard location first, then look in root_dir
                standard_imagenet = os.environ.get('IMAGENET_ROOT', os.path.join(root_dir, 'imagenet'))
                if os.path.exists(standard_imagenet):
                    imagenet_path = os.path.join(standard_imagenet, split_dir)
                else:
                    imagenet_path = os.path.join(root_dir, 'imagenet', split_dir)
            
            self.base_dataset = datasets.ImageFolder(
                root=imagenet_path,
                transform=transform
            )
        elif self.dataset_name == 'cocologic10':
            # CoCoLogic uses custom dataset
            if not COCOLOGIC_AVAILABLE:
                raise ImportError("CoCoLogic dataset not available. Check generate_COCOLogic.py")
            
            # Handle different root_dir formats for CoCoLogic
            # Case 1: root_dir already points to coco/data (e.g., /path/to/coco/data)
            # Case 2: root_dir is base path and we need to append coco (e.g., /path/to/datasets)
            if 'coco' in root_dir.lower() and root_dir.endswith('data'):
                # Already pointing to coco/data directory
                coco_root = root_dir
            else:
                # Need to append coco subdirectory
                coco_root = os.path.join(root_dir, 'coco')
            
            split_name = 'train2017' if train else 'val2017'
            annotation_file = os.path.join(coco_root, 'annotations', f'instances_{split_name}.json')
            image_dir = os.path.join(coco_root, split_name)
            
            category_map = load_category_mapping(annotation_file)
            self.base_dataset = COCOLogicDataset(
                annotation_file=annotation_file,
                image_dir=image_dir,
                category_id_to_name=category_map,
                transform=transform,
                filter_no_labels=True,
                exclusive_label=True,
                exclusive_match_only=True,
                log_statistics=False,
                version=10
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from: cifar100, imagenet, cocologic10")
        
        # Load precomputed concept similarities from HDF5
        if embeddings_path is None:
            raise ValueError("embeddings_path must be provided")
        
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        with h5py.File(embeddings_path, 'r') as f:
            # Load embeddings and labels
            split_prefix = 'train' if train else 'test'
            self.concept_similarities = f[f'{split_prefix}_embeddings'][:]
            self.labels = f[f'{split_prefix}_labels'][:]
        
        # Normalize concept similarities from [-1, 1] to [0, 1]
        self.concept_truth = (self.concept_similarities + 1.0) / 2.0
        
        # Verify alignment by checking labels match
        print(f"\nVerifying dataset alignment for {self.dataset_name} {'train' if train else 'test'} split...")
        mismatches = 0
        for i in range(min(100, len(self.base_dataset))):  # Check first 100 samples
            _, dataset_label = self.base_dataset[i]
            if dataset_label != self.labels[i]:
                mismatches += 1
        
        if mismatches > 0:
            print(f"WARNING: Found {mismatches} label mismatches in first 100 samples!")
        else:
            print(f"✓ Label alignment verified for {self.dataset_name} {'train' if train else 'test'} split")
        
        print(f"Loaded {len(self.base_dataset)} samples with {self.concept_truth.shape[1]} concepts")
        print(f"Concept truth range: [{self.concept_truth.min():.3f}, {self.concept_truth.max():.3f}]")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Preprocessed image tensor
            label: Class label
            concept_truth: Normalized concept similarities [0, 1]^(n_concepts)
        """
        image, label = self.base_dataset[idx]
        concept_truth = torch.from_numpy(self.concept_truth[idx]).float()
        return image, label, concept_truth


class DCRModel(nn.Module):
    """
    Deep Concept Reasoning model for image classification.
    
    Supports: CIFAR-100 (100 classes), ImageNet (1000 classes), CoCoLogic10 (10 classes)
    
    Architecture:
        image → ResNet-18 backbone → ConceptEmbedding → ConceptReasoningLayer → logits
    """
    
    def __init__(self, n_concepts=10000, emb_dim=8, n_classes=100, hidden_dim=512):
        """
        Args:
            n_concepts: Number of concepts (vocab size)
            emb_dim: Embedding dimension for concept reasoning
            n_classes: Number of output classes
            hidden_dim: Hidden dimension from backbone (512 for ResNet-18)
        """
        super().__init__()
        
        # ResNet-18 backbone (remove final FC layer)
        resnet = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.backbone_dim = 512  # ResNet-18 outputs 512-dim features
        
        # Concept embedding layer (outputs c_emb and c_pred)
        self.concept_embedding = ConceptEmbedding(
            in_features=self.backbone_dim,
            n_concepts=n_concepts,
            emb_size=emb_dim
        )
        
        # Concept reasoning layer for classification
        self.task_predictor = ConceptReasoningLayer(
            emb_size=emb_dim,
            n_classes=n_classes
        )
    
    def forward(self, x):
        """
        Args:
            x: Input images [batch_size, 3, 32, 32]
        
        Returns:
            y_pred: Class logits [batch_size, n_classes]
            c_pred: Concept predictions [batch_size, n_concepts]
            c_emb: Concept embeddings [batch_size, n_concepts, emb_dim]
        """
        # Extract features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Get concept embeddings and predictions
        c_emb, c_pred = self.concept_embedding(features)
        
        # Get class predictions from concept reasoning
        y_pred = self.task_predictor(c_emb, c_pred)
        
        return y_pred, c_pred, c_emb


def train_epoch(model, train_loader, optimizer, device, lambda_task=1.0, num_classes=100):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    concept_loss_sum = 0
    task_loss_sum = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels, concept_truth in pbar:
        images = images.to(device)
        labels = labels.to(device)
        concept_truth = concept_truth.to(device)
        
        # Forward pass
        y_pred, c_pred, c_emb = model(images)
        
        # Compute concept loss (BCE between predicted and ground-truth concepts)
        # Note: c_pred is already sigmoid-activated by ConceptEmbedding, so use BCE not BCE-with-logits
        concept_loss = F.binary_cross_entropy(c_pred, concept_truth)
        
        # Compute task loss (BCE for classification with one-hot encoding)
        # Note: ConceptReasoningLayer outputs probabilities (after sigmoid), so use BCE not CE
        y_onehot = F.one_hot(labels, num_classes=num_classes).float()
        task_loss = F.binary_cross_entropy(y_pred, y_onehot)
        
        # Combined loss
        loss = concept_loss + lambda_task * task_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        concept_loss_sum += concept_loss.item()
        task_loss_sum += task_loss.item()
        
        _, predicted = y_pred.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'c_loss': f'{concept_loss.item():.4f}',
            't_loss': f'{task_loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_concept_loss = concept_loss_sum / len(train_loader)
    avg_task_loss = task_loss_sum / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_concept_loss, avg_task_loss, accuracy


def evaluate(model, test_loader, device, dataset_name='cifar100'):
    """Evaluate on test set."""
    model.eval()
    correct = 0
    total = 0
    all_c_emb = []
    all_c_pred = []
    
    # For CoCoLogic: collect predictions for balanced accuracy
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, concept_truth in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            y_pred, c_pred, c_emb = model(images)
            
            _, predicted = y_pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Collect for balanced accuracy (CoCoLogic)
            if dataset_name.lower() == 'cocologic10':
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
            
            # Store for rule extraction (only first batch to save memory)
            if len(all_c_emb) == 0:
                all_c_emb.append(c_emb.cpu())
                all_c_pred.append(c_pred.cpu())
    
    # Calculate accuracy
    accuracy = 100. * correct / total
    
    # For CoCoLogic, also calculate balanced accuracy
    balanced_acc = None
    if dataset_name.lower() == 'cocologic10':
        balanced_acc = 100. * balanced_accuracy_score(all_labels, all_predictions)
    
    # Concatenate first batch for rule extraction
    c_emb_sample = all_c_emb[0] if all_c_emb else None
    c_pred_sample = all_c_pred[0] if all_c_pred else None
    
    return accuracy, balanced_acc, c_emb_sample, c_pred_sample


def extract_rules(model, c_emb, c_pred):
    """Extract logical rules from the trained model."""
    print("\n" + "="*80)
    print("EXTRACTING GLOBAL LOGICAL RULES")
    print("="*80)
    
    try:
        # Get global explanations
        explanations = model.task_predictor.explain(
            c_emb, 
            c_pred, 
            mode='global'
        )
        
        print("\nGlobal logical rules extracted from DCR model:")
        print(explanations)
        
    except Exception as e:
        print(f"\nWarning: Could not extract rules: {e}")
        print("This is normal if the model hasn't converged or if torch-explain doesn't support this configuration.")


def main():
    parser = argparse.ArgumentParser(description='Train DCR on CIFAR-100, ImageNet, or CoCoLogic10')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'imagenet', 'cocologic10'],
                        help='Dataset to use')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory for embeddings (e.g., ./data or /path/to/datasets)')
    parser.add_argument('--image_root_dir', type=str, default=None,
                        help='Root directory for images if different from root_dir (optional, for ImageNet/CoCoLogic)')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Vocabulary size (number of concepts)')
    
    # Model arguments
    parser.add_argument('--emb_dim', type=int, default=8,
                        help='Concept embedding dimension')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lambda_task', type=float, default=1.0,
                        help='Weight for task loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Initialize WandB if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="Concept-Learning",
            config={
                "dataset": args.dataset,
                "vocab_size": args.vocab_size,
                "emb_dim": args.emb_dim,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "lambda_task": args.lambda_task,
            }
        )
    elif args.wandb and not WANDB_AVAILABLE:
        print("Warning: --wandb flag set but wandb is not installed")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Set number of classes based on dataset
    dataset_name = args.dataset.lower()
    if dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'imagenet':
        num_classes = 1000
    elif dataset_name == 'cocologic10':
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Construct embeddings path (following trainer_framework.py pattern)
    embeddings_dir = os.path.join(
        args.root_dir,
        f'embeddings_{dataset_name}',
        f'vocab_{args.vocab_size}',
        'nonsparse'
    )
    embeddings_path = os.path.join(embeddings_dir, f'{dataset_name}_splice_embeddings_full.h5')
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Root directory: {args.root_dir}")
    print(f"  Embeddings path: {embeddings_path}")
    print(f"  Vocabulary size: {args.vocab_size}")
    print(f"  Embedding dimension: {args.emb_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Lambda (task weight): {args.lambda_task}")
    print(f"  Epochs: {args.epochs}")
    
    # Define transforms based on dataset
    if dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                               std=[0.2675, 0.2565, 0.2761])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                               std=[0.2675, 0.2565, 0.2761])
        ])
    elif dataset_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    elif dataset_name == 'cocologic10':
        # CoCoLogic uses similar preprocessing to ImageNet
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Determine image root directory
    # For ImageNet and CoCoLogic, images might be at a different location than embeddings
    if args.image_root_dir:
        image_root = args.image_root_dir
    elif dataset_name == 'imagenet':
        # Default ImageNet location (override with --image_root_dir or $IMAGENET_ROOT)
        image_root = os.environ.get('IMAGENET_ROOT', './data/imagenet')
    elif dataset_name == 'cocologic10':
        # Default CoCoLogic location (override with --image_root_dir or $COCO_ROOT)
        image_root = os.environ.get('COCO_ROOT', './data/coco/data')
    else:
        # CIFAR-100 and others use same root as embeddings
        image_root = args.root_dir
    
    print(f"  Image root directory: {image_root}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = DatasetWithConcepts(
        root_dir=image_root,
        dataset_name=dataset_name,
        train=True,
        embeddings_path=embeddings_path,
        transform=transform_train
    )
    
    test_dataset = DatasetWithConcepts(
        root_dir=image_root,
        dataset_name=dataset_name,
        train=False,
        embeddings_path=embeddings_path,
        transform=transform_test
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle during training
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating DCR model...")
    model = DCRModel(
        n_concepts=args.vocab_size,
        emb_dim=args.emb_dim,
        n_classes=num_classes,
        hidden_dim=512
    ).to(device)
    
    print(f"Model architecture:")
    print(f"  Backbone: ResNet-18 (512-dim features)")
    print(f"  ConceptEmbedding: {512} → {args.vocab_size} concepts × {args.emb_dim} dims")
    print(f"  ConceptReasoningLayer: {args.emb_dim} dims → {num_classes} classes")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_c_loss, train_t_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, args.lambda_task, num_classes
        )
        
        # Evaluate
        test_acc, test_balanced_acc, c_emb_sample, c_pred_sample = evaluate(
            model, test_loader, device, dataset_name
        )
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (Concept: {train_c_loss:.4f}, Task: {train_t_loss:.4f})")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        
        # For CoCoLogic, also print balanced accuracy
        if dataset_name == 'cocologic10' and test_balanced_acc is not None:
            print(f"  Test Balanced Accuracy: {test_balanced_acc:.2f}%")
        
        # Log to WandB
        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/concept_loss": train_c_loss,
                "train/task_loss": train_t_loss,
                "train/accuracy": train_acc,
                "test_accuracy": test_acc,
            }
            
            # Add balanced accuracy for CoCoLogic and track best appropriately
            if dataset_name == 'cocologic10' and test_balanced_acc is not None:
                log_dict["test_balanced_accuracy"] = test_balanced_acc
                log_dict["best_balanced_accuracy"] = max(best_acc, test_balanced_acc)
            else:
                log_dict["best_accuracy"] = max(best_acc, test_acc)
            
            wandb.log(log_dict)
        
        # Track best accuracy (use balanced accuracy for CoCoLogic)
        current_acc = test_balanced_acc if (dataset_name == 'cocologic10' and test_balanced_acc is not None) else test_acc
        if current_acc > best_acc:
            best_acc = current_acc
            metric_name = "balanced accuracy" if dataset_name == 'cocologic10' else "accuracy"
            print(f"  ✓ New best {metric_name}: {best_acc:.2f}%")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    
    # Print best metric (balanced accuracy for CoCoLogic, regular accuracy for others)
    if dataset_name == 'cocologic10':
        print(f"\nBest Test Balanced Accuracy: {best_acc:.2f}%")
    else:
        print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    
    # Log final summary to WandB
    if use_wandb:
        if dataset_name == 'cocologic10':
            wandb.summary["best_test_balanced_accuracy"] = best_acc
        else:
            wandb.summary["best_test_accuracy"] = best_acc
        wandb.finish()
    
    # Extract rules from trained model
    if c_emb_sample is not None and c_pred_sample is not None:
        extract_rules(model, c_emb_sample.to(device), c_pred_sample.to(device))
    else:
        print("\nWarning: Could not extract rules (no samples collected)")


if __name__ == '__main__':
    main()

