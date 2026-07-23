#!/usr/bin/env python
# filepath: src/analysis/analyze_masks.py

import os
import argparse
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze feature masks from SpLiCE')
    parser.add_argument('--masks_path', type=str, required=True,
                        help='Path to the saved masks HDF5 file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the original embeddings HDF5 file (for class labels)')
    parser.add_argument('--output_dir', type=str, default='mask_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Show top K most important features per class')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'imagenet'],
                        help='Dataset name for analysis')
    parser.add_argument('--plot_heatmap', action='store_true',
                        help='Generate heatmap visualizations of feature importance')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for processing data')
    parser.add_argument('--compute_precision_entropy', action='store_true',
                        help='Compute average precision and conditional entropy for masks')
    parser.add_argument('--selector', type=str, default='merlin', choices=['merlin','morgana'],
                        help='Which selector masks to use when computing precision/entropy')
    parser.add_argument('--tolerance', type=int, default=0,
                        help='Hamming-distance tolerance to count a match')
    parser.add_argument('--target_class', type=int, default=None,
                        help='If set compute metrics for only this class; otherwise for all classes')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed used for inner shuffling')
    parser.add_argument('--compute_only', action='store_true',
                        help='If set, run only the precision/entropy computation and exit')
    parser.add_argument('--n_clusters', type=int, default=50,
                        help='Number of clusters for mask clustering analysis')
    parser.add_argument('--cluster_analysis', action='store_true',
                        help='If set, perform cluster-based analysis of masks')
    return parser.parse_args()


def compute_precision_and_entropy_masks(data: Dict,
                                        selector: str = 'merlin',
                                        target_class: Optional[int] = None,
                                        tolerance: int = 0,
                                        batch_size: int = 1024,
                                        seed: int = 42,
                                        device: str = 'cpu'):
    """
    Compute average precision and conditional entropy using mask similarity (Hamming distance).
    Returns dict mapping class -> (average_precision, conditional_entropy)
    """
    assert selector in ('merlin', 'morgana')
    key = f"val_{selector}_masks"
    if key not in data:
        raise KeyError(f"Data does not contain '{key}'")

    masks = data[key].to(torch.uint8)    # (N, D) 0/1
    labels = data['val_labels'].long() # (N,)
    num_samples = masks.shape[0]

    # determine number of classes (assumes classes are 0..C-1)
    num_classes = int(labels.max().item()) + 1

    # prepare list of target classes
    classes = list(range(num_classes))
    if target_class is not None:
        classes = [int(target_class)]

    # Use optimized exact matching for tolerance=0
    if tolerance == 0:
        return _compute_precision_entropy_exact_matches(masks, labels, classes, num_classes)
    else:
        return _compute_precision_entropy_hamming(masks, labels, classes, num_classes, tolerance, batch_size, seed)


def _compute_precision_entropy_exact_matches(masks, labels, classes, num_classes):
    """Optimized version for exact matches only (tolerance=0)"""
    print("Using optimized exact matching algorithm...")
    
    # Convert masks to hashable tuples for fast lookup
    print("Building mask index...")
    mask_to_indices = defaultdict(list)
    
    for i in tqdm(range(len(masks)), desc="Indexing masks"):
        mask_tuple = tuple(masks[i].cpu().numpy().astype(np.uint8))
        mask_to_indices[mask_tuple].append(i)
    
    print(f"Found {len(mask_to_indices)} unique mask patterns")
    
    results = {}
    class_pbar = tqdm(classes, desc="Processing classes (exact)")
    
    for cls in class_pbar:
        class_pbar.set_description(f"Processing class {cls} (exact)")
        
        # indices of target-class samples
        target_indices = (labels == cls).nonzero(as_tuple=True)[0]
        if target_indices.numel() == 0:
            results[cls] = (float('nan'), float('nan'), float('nan'))
            continue

        average_precision_list = []
        conditional_entropy_list = []
        occurrence_count_list = []
        
        sample_pbar = tqdm(target_indices, desc=f"Class {cls} samples", leave=False)
        
        for ti in sample_pbar:
            target_mask = tuple(masks[ti].cpu().numpy().astype(np.uint8))
            
            # Find all samples with identical mask (O(1) lookup!)
            matching_indices = mask_to_indices[target_mask]
            occurrence_count = len(matching_indices)
            occurrence_count_list.append(occurrence_count)
            
            if occurrence_count == 0:
                average_precision_list.append(0.0)
                conditional_entropy_list.append(0.0)
                continue
            
            # Count classes of matching samples
            matching_labels = labels[matching_indices]
            class_counts = torch.bincount(matching_labels, minlength=num_classes).float()
            
            total = class_counts.sum().item()
            if total == 0:
                class_counts += 1e-6
                total = class_counts.sum().item()
            
            class_probs = class_counts / total
            
            # Average precision = probability mass on true class
            true_prob = class_probs[cls].item()
            average_precision_list.append(true_prob)
            
            # Conditional entropy: -sum p log p (natural log)
            p_mask = class_probs > 0
            cond_ent = -(class_probs[p_mask] * torch.log(class_probs[p_mask])).sum().item()
            conditional_entropy_list.append(cond_ent)
            
            # Update progress
            if len(average_precision_list) > 0:
                avg_prec_so_far = sum(average_precision_list) / len(average_precision_list)
                sample_pbar.set_postfix(avg_precision=f"{avg_prec_so_far:.3f}")

        avg_prec = float(torch.tensor(average_precision_list).mean().item())
        avg_ent = float(torch.tensor(conditional_entropy_list).mean().item())
        avg_occurrence = float(torch.tensor(occurrence_count_list, dtype=torch.float32).mean().item())
        results[cls] = (avg_prec, avg_ent, avg_occurrence)
        
        class_pbar.set_postfix(precision=f"{avg_prec:.3f}", entropy=f"{avg_ent:.3f}", avg_occurrence=f"{avg_occurrence:.2f}")

    return results


def _compute_precision_entropy_hamming(masks, labels, classes, num_classes, tolerance, batch_size, seed):
    """Original Hamming distance version for tolerance > 0"""
    print(f"Using Hamming distance algorithm with tolerance={tolerance}...")
    
    # shuffled inner order for comparisons
    rng = torch.Generator()
    rng.manual_seed(seed)
    shuffled_idx = torch.randperm(len(masks), generator=rng)
    masks_shuffled = masks[shuffled_idx]
    labels_shuffled = labels[shuffled_idx]
    
    results = {}
    class_pbar = tqdm(classes, desc=f"Processing classes (hamming)")
    
    for cls in class_pbar:
        class_pbar.set_description(f"Processing class {cls} (hamming)")
        
        # indices of target-class samples in original order
        target_indices = (labels == cls).nonzero(as_tuple=True)[0]
        if target_indices.numel() == 0:
            results[cls] = (float('nan'), float('nan'))
            continue

        average_precision_list = []
        conditional_entropy_list = []

        # Progress bar for samples within this class
        sample_pbar = tqdm(target_indices, desc=f"Class {cls} samples", leave=False)
        
        for ti in sample_pbar:
            target_mask = masks[ti].unsqueeze(0)            # (1, D)
            occurrence_per_class = torch.zeros(num_classes, dtype=torch.float32)

            # iterate inner dataset in chunks
            for start in range(0, len(masks), batch_size):
                end = min(start + batch_size, len(masks))
                inner_masks = masks_shuffled[start:end]      # (B, D)
                inner_labels = labels_shuffled[start:end]    # (B,)

                # Hamming distance (number of differing bits)
                xor = target_mask != inner_masks             # (B, D) bool
                dist = xor.sum(dim=1)                        # (B,)

                matches = dist <= tolerance
                if matches.any():
                    matched_labels = inner_labels[matches]
                    # bincount to get counts per class (vectorized)
                    counts = torch.bincount(matched_labels, minlength=num_classes).float()
                    occurrence_per_class += counts

            total = occurrence_per_class.sum().item()
            if total == 0:
                # avoid div by zero: treat as tiny total -> uniform-like tiny prob
                occurrence_per_class += 1e-6
                total = occurrence_per_class.sum().item()

            class_probs = occurrence_per_class / total

            # average precision = probability mass on true class
            true_prob = class_probs[cls].item()
            average_precision_list.append(true_prob)

            # conditional entropy: -sum p log p (natural log)
            p = class_probs
            mask = p > 0
            cond_ent = - (p[mask] * torch.log(p[mask])).sum().item()
            conditional_entropy_list.append(cond_ent)
            
            # Update sample progress bar with current precision
            if len(average_precision_list) > 0:
                avg_prec_so_far = sum(average_precision_list) / len(average_precision_list)
                sample_pbar.set_postfix(avg_precision=f"{avg_prec_so_far:.3f}")

        avg_prec = float(torch.tensor(average_precision_list).mean().item())
        avg_ent = float(torch.tensor(conditional_entropy_list).mean().item())
        results[cls] = (avg_prec, avg_ent)
        
        # Update class progress bar with results
        class_pbar.set_postfix(precision=f"{avg_prec:.3f}", entropy=f"{avg_ent:.3f}")

    return results

def load_masks_and_data(masks_path: str, data_path: str, batch_size: int = 1024):
    """
    Load masks and corresponding data embeddings with labels
    """
    print(f"Loading masks from {masks_path}")
    with h5py.File(masks_path, 'r') as f:
        # Get metadata
        metadata = dict(f['metadata'].attrs)
        mask_size = metadata.get('mask_size', 'Unknown')
        embedding_dim = metadata.get('embedding_dim', 'Unknown')
        completeness = metadata.get('val_completeness', 'Unknown')
        soundness = metadata.get('val_soundness', 'Unknown')
        
        print(f"Mask metadata:")
        print(f"  Mask size: {mask_size}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Validation completeness: {completeness}")
        print(f"  Validation soundness: {soundness}")
        
        # Load masks in batches to handle large datasets
        train_merlin_masks = []
        train_morgana_masks = []
        val_merlin_masks = []
        val_morgana_masks = []
        
        # Training masks
        train_size = f['train_merlin_masks'].shape[0]
        for i in tqdm(range(0, train_size, batch_size), desc="Loading training masks"):
            end = min(i + batch_size, train_size)
            train_merlin_masks.append(torch.from_numpy(f['train_merlin_masks'][i:end]))
            train_morgana_masks.append(torch.from_numpy(f['train_morgana_masks'][i:end]))
        
        train_merlin_masks = torch.cat(train_merlin_masks, dim=0)
        train_morgana_masks = torch.cat(train_morgana_masks, dim=0)
        
        # Validation masks (if available)
        if 'val_merlin_masks' in f and 'val_morgana_masks' in f:
            val_size = f['val_merlin_masks'].shape[0]
            for i in tqdm(range(0, val_size, batch_size), desc="Loading validation masks"):
                end = min(i + batch_size, val_size)
                val_merlin_masks.append(torch.from_numpy(f['val_merlin_masks'][i:end]))
                val_morgana_masks.append(torch.from_numpy(f['val_morgana_masks'][i:end]))
            
            val_merlin_masks = torch.cat(val_merlin_masks, dim=0)
            val_morgana_masks = torch.cat(val_morgana_masks, dim=0)
    
    print(f"Loading data from {data_path}")
    with h5py.File(data_path, 'r') as f:
        # Load labels in batches to handle large datasets
        train_labels = []
        val_labels = []
        
        train_size = f['train_labels'].shape[0]
        for i in tqdm(range(0, train_size, batch_size), desc="Loading training labels"):
            end = min(i + batch_size, train_size)
            train_labels.append(torch.from_numpy(f['train_labels'][i:end]))
        
        train_labels = torch.cat(train_labels, dim=0).long()
        
        val_size = f['test_labels'].shape[0]
        for i in tqdm(range(0, val_size, batch_size), desc="Loading validation labels"):
            end = min(i + batch_size, val_size)
            val_labels.append(torch.from_numpy(f['test_labels'][i:end]))
        
        val_labels = torch.cat(val_labels, dim=0).long()
    
    return {
        'train_merlin_masks': train_merlin_masks,
        'train_morgana_masks': train_morgana_masks,
        'val_merlin_masks': val_merlin_masks if len(val_merlin_masks) > 0 else None,
        'val_morgana_masks': val_morgana_masks if len(val_morgana_masks) > 0 else None,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'metadata': metadata
    }

def analyze_general_stats(data: Dict) -> Dict:
    """
    Analyze general statistics for masks
    """
    results = {}
    
    # Training stats
    train_merlin = data['train_merlin_masks']
    train_morgana = data['train_morgana_masks']
    
    # Basic statistics
    results['train_merlin_avg_features'] = train_merlin.sum(1).float().mean().item()
    results['train_merlin_std_features'] = train_merlin.sum(1).float().std().item()
    results['train_merlin_min_features'] = train_merlin.sum(1).min().item()
    results['train_merlin_max_features'] = train_merlin.sum(1).max().item()
    
    results['train_morgana_avg_features'] = train_morgana.sum(1).float().mean().item()
    results['train_morgana_std_features'] = train_morgana.sum(1).float().std().item()
    results['train_morgana_min_features'] = train_morgana.sum(1).min().item()
    results['train_morgana_max_features'] = train_morgana.sum(1).max().item()
    
    # Feature activation frequencies
    results['train_merlin_feature_freqs'] = train_merlin.sum(0).float() / train_merlin.shape[0]
    results['train_morgana_feature_freqs'] = train_morgana.sum(0).float() / train_morgana.shape[0]
    
    # Overlap between Merlin and Morgana
    overlap_count = ((train_merlin > 0) & (train_morgana > 0)).sum(1).float()
    results['train_mask_overlap_avg'] = overlap_count.mean().item()
    results['train_mask_overlap_std'] = overlap_count.std().item()
    results['train_mask_overlap_pct'] = 100 * overlap_count.mean().item() / results['train_merlin_avg_features']
    
    # Top activated features
    results['train_merlin_top_features'] = torch.argsort(results['train_merlin_feature_freqs'], descending=True)[:50].tolist()
    results['train_morgana_top_features'] = torch.argsort(results['train_morgana_feature_freqs'], descending=True)[:50].tolist()
    
    # Similar stats for validation if available
    if data['val_merlin_masks'] is not None:
        val_merlin = data['val_merlin_masks']
        val_morgana = data['val_morgana_masks']
        
        results['val_merlin_avg_features'] = val_merlin.sum(1).float().mean().item()
        results['val_merlin_std_features'] = val_merlin.sum(1).float().std().item()
        results['val_merlin_min_features'] = val_merlin.sum(1).min().item()
        results['val_merlin_max_features'] = val_merlin.sum(1).max().item()
        
        results['val_morgana_avg_features'] = val_morgana.sum(1).float().mean().item()
        results['val_morgana_std_features'] = val_morgana.sum(1).float().std().item()
        results['val_morgana_min_features'] = val_morgana.sum(1).min().item()
        results['val_morgana_max_features'] = val_morgana.sum(1).max().item()
        
        results['val_merlin_feature_freqs'] = val_merlin.sum(0).float() / val_merlin.shape[0]
        results['val_morgana_feature_freqs'] = val_morgana.sum(0).float() / val_morgana.shape[0]
        
        val_overlap_count = ((val_merlin > 0) & (val_morgana > 0)).sum(1).float()
        results['val_mask_overlap_avg'] = val_overlap_count.mean().item()
        results['val_mask_overlap_std'] = val_overlap_count.std().item()
        results['val_mask_overlap_pct'] = 100 * val_overlap_count.mean().item() / results['val_merlin_avg_features']
    
    return results

def analyze_per_class_stats(data: Dict, top_k: int = 10) -> Dict:
    """
    Analyze per-class feature importance statistics
    """
    results = {'train': {}, 'val': {}}
    
    # Get unique classes
    classes = torch.unique(data['train_labels']).tolist()
    
    # Training data analysis
    train_merlin = data['train_merlin_masks']
    train_morgana = data['train_morgana_masks']
    train_labels = data['train_labels']
    
    # Per-class feature importance
    for cls in tqdm(classes, desc="Analyzing per-class stats (train)"):
        cls_idx = train_labels == cls
        if cls_idx.sum() == 0:
            continue
            
        # Get masks for this class
        cls_merlin_masks = train_merlin[cls_idx]
        cls_morgana_masks = train_morgana[cls_idx]
        
        # Compute feature frequencies for this class
        cls_merlin_feats = cls_merlin_masks.sum(0).float() / cls_idx.sum()
        cls_morgana_feats = cls_morgana_masks.sum(0).float() / cls_idx.sum()
        
        # Store statistics
        results['train'][cls] = {
            'count': cls_idx.sum().item(),
            'merlin_avg_features': cls_merlin_masks.sum(1).float().mean().item(),
            'merlin_std_features': cls_merlin_masks.sum(1).float().std().item(),
            'morgana_avg_features': cls_morgana_masks.sum(1).float().mean().item(),
            'morgana_std_features': cls_morgana_masks.sum(1).float().std().item(),
            'merlin_feature_freqs': cls_merlin_feats,
            'morgana_feature_freqs': cls_morgana_feats,
            'merlin_top_features': torch.argsort(cls_merlin_feats, descending=True)[:top_k].tolist(),
            'merlin_top_values': torch.sort(cls_merlin_feats, descending=True)[0][:top_k].tolist(),
            'morgana_top_features': torch.argsort(cls_morgana_feats, descending=True)[:top_k].tolist(),
            'morgana_top_values': torch.sort(cls_morgana_feats, descending=True)[0][:top_k].tolist(),
        }
    
    # Validation data analysis if available
    if data['val_merlin_masks'] is not None:
        val_merlin = data['val_merlin_masks']
        val_morgana = data['val_morgana_masks']
        val_labels = data['val_labels']
        
        for cls in tqdm(classes, desc="Analyzing per-class stats (val)"):
            cls_idx = val_labels == cls
            if cls_idx.sum() == 0:
                continue
                
            # Get masks for this class
            cls_merlin_masks = val_merlin[cls_idx]
            cls_morgana_masks = val_morgana[cls_idx]
            
            # Compute feature frequencies for this class
            cls_merlin_feats = cls_merlin_masks.sum(0).float() / cls_idx.sum()
            cls_morgana_feats = cls_morgana_masks.sum(0).float() / cls_idx.sum()
            
            # Store statistics
            results['val'][cls] = {
                'count': cls_idx.sum().item(),
                'merlin_avg_features': cls_merlin_masks.sum(1).float().mean().item(),
                'merlin_std_features': cls_merlin_masks.sum(1).float().std().item(),
                'morgana_avg_features': cls_morgana_masks.sum(1).float().mean().item(),
                'morgana_std_features': cls_morgana_masks.sum(1).float().std().item(),
                'merlin_feature_freqs': cls_merlin_feats,
                'morgana_feature_freqs': cls_morgana_feats,
                'merlin_top_features': torch.argsort(cls_merlin_feats, descending=True)[:top_k].tolist(),
                'merlin_top_values': torch.sort(cls_merlin_feats, descending=True)[0][:top_k].tolist(),
                'morgana_top_features': torch.argsort(cls_morgana_feats, descending=True)[:top_k].tolist(),
                'morgana_top_values': torch.sort(cls_morgana_feats, descending=True)[0][:top_k].tolist(),
            }
    
    # Analyze feature uniqueness per class
    train_unique_features = analyze_unique_features(results['train'], top_k=top_k)
    results['train_unique_features'] = train_unique_features
    
    if 'val' in results and results['val']:
        val_unique_features = analyze_unique_features(results['val'], top_k=top_k)
        results['val_unique_features'] = val_unique_features
    
    return results

def analyze_unique_features(class_stats: Dict, top_k: int = 10) -> Dict:
    """
    Analyze which features are uniquely important for each class
    """
    unique_features = {}
    all_classes = list(class_stats.keys())
    
    # For each class, find features that are in its top-k but not in other classes' top-k
    for cls in all_classes:
        merlin_top_features = set(class_stats[cls]['merlin_top_features'])
        other_top_features = set()
        
        for other_cls in all_classes:
            if other_cls != cls:
                other_top_features.update(class_stats[other_cls]['merlin_top_features'])
        
        # Find unique features for this class
        unique = merlin_top_features - other_top_features
        unique_features[cls] = {
            'unique_count': len(unique),
            'unique_features': list(unique),
            'unique_pct': 100 * len(unique) / len(merlin_top_features) if merlin_top_features else 0
        }
    
    return unique_features

def generate_visualizations(results: Dict, output_dir: str, dataset_name: str, plot_heatmap: bool):
    """
    Generate visualizations for mask analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results
    general_stats = results['general_stats']
    class_stats = results.get('class_stats', {})
    
    # Get the feature frequencies
    merlin_freqs = general_stats['train_merlin_feature_freqs'].numpy()
    morgana_freqs = general_stats['train_morgana_feature_freqs'].numpy()
    
    # Plot basic feature frequency distribution
    plt.figure(figsize=(12, 6))
    plt.hist(merlin_freqs, bins=50, alpha=0.7, label='Merlin')
    plt.hist(morgana_freqs, bins=50, alpha=0.7, label='Morgana')
    plt.title('Feature Frequency Distribution')
    plt.xlabel('Selection Frequency')
    plt.ylabel('Number of Features')
    plt.legend()
    plt.yscale('log')  # Log scale for better visibility
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_frequency_distribution.png'), dpi=300)
    plt.close()
    
    # ===== TOP MERLIN FEATURES =====
    top_features = 50  # Number of top features to show
    
    # Find the indices of the top features
    top_merlin_indices = np.argsort(merlin_freqs)[::-1][:top_features]
    
    plt.figure(figsize=(15, 6))
    plt.bar(np.arange(top_features), merlin_freqs[top_merlin_indices] * 100, 
            color='blue', alpha=0.7)
    
    # Add feature indices as x-tick labels
    plt.xticks(np.arange(top_features), top_merlin_indices, rotation=90, fontsize=8)
    plt.xlabel('Feature Index')
    plt.ylabel('Usage Percentage (%)')
    plt.title(f'Top {top_features} Most Frequently Used Features (Merlin)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_merlin_feature_percentages.png'), dpi=300)
    plt.close()
    
    # ===== TOP MORGANA FEATURES =====
    top_morgana_indices = np.argsort(morgana_freqs)[::-1][:top_features]
    
    plt.figure(figsize=(15, 6))
    plt.bar(np.arange(top_features), morgana_freqs[top_morgana_indices] * 100, 
            color='red', alpha=0.7)
    
    # Add feature indices as x-tick labels
    plt.xticks(np.arange(top_features), top_morgana_indices, rotation=90, fontsize=8)
    plt.xlabel('Feature Index')
    plt.ylabel('Usage Percentage (%)')
    plt.title(f'Top {top_features} Most Frequently Used Features (Morgana)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_morgana_feature_percentages.png'), dpi=300)
    plt.close()
    
    # ===== FEATURE USAGE DISTRIBUTION =====
    # Histogram bins showing distribution of feature usage percentages
    plt.figure(figsize=(12, 6))
    
    # Create histogram bins
    bin_edges = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    bin_labels = ['0-1%', '1-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-40%', '40-50%', '50-75%', '75-100%']
    
    # Count features in each bin
    merlin_hist, _ = np.histogram(merlin_freqs, bins=bin_edges)
    morgana_hist, _ = np.histogram(morgana_freqs, bins=bin_edges)
    
    # Plot side by side
    x = np.arange(len(bin_labels))
    width = 0.35
    
    plt.bar(x - width/2, merlin_hist, width, label='Merlin', color='blue', alpha=0.7)
    plt.bar(x + width/2, morgana_hist, width, label='Morgana', color='red', alpha=0.7)
    
    plt.xlabel('Feature Usage Percentage')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Usage Percentages')
    plt.xticks(x, bin_labels, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_usage_distribution.png'), dpi=300)
    plt.close()
    
    # ===== UNIQUE FEATURES PER CLASS =====
    if 'train_unique_features' in class_stats:
        unique_features = class_stats['train_unique_features']
        classes = sorted(unique_features.keys())
        unique_counts = [unique_features[c]['unique_count'] for c in classes]
        unique_pcts = [unique_features[c]['unique_pct'] for c in classes]
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.bar(classes, unique_counts)
        plt.xlabel('Class')
        plt.ylabel('Number of Unique Features')
        plt.title('Number of Class-Specific Unique Features')
        plt.xticks(rotation=90)
        
        plt.subplot(2, 1, 2)
        plt.bar(classes, unique_pcts)
        plt.xlabel('Class')
        plt.ylabel('Percentage of Unique Features')
        plt.title('Percentage of Class-Specific Unique Features')
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'unique_features_per_class.png'), dpi=300)
        plt.close()
    
    # ===== CLASS FEATURE CORRELATION HEATMAP =====
    if plot_heatmap and 'train_class_feature_correlation' in class_stats:
        corr_matrix = class_stats['train_class_feature_correlation']
        plt.figure(figsize=(12, 10))
        plt.imshow(corr_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Correlation')
        plt.title('Class Feature Correlation Matrix')
        plt.xlabel('Class')
        plt.ylabel('Class')
        
        # Add gridlines
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_feature_correlation.png'), dpi=300)
        plt.close()
    
    # ===== FEATURE PERCENTAGES CSV =====
    # Export raw feature percentages to CSV for further analysis
    with open(os.path.join(output_dir, 'feature_percentages.csv'), 'w') as f:
        f.write('feature_idx,merlin_percentage,morgana_percentage\n')
        for i in range(len(merlin_freqs)):
            f.write(f'{i},{merlin_freqs[i]*100:.2f},{morgana_freqs[i]*100:.2f}\n')

def generate_reports(results: Dict, output_dir: str, dataset_name: str, top_k: int):
    """
    Generate text reports summarizing the mask analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # General summary report
    general_stats = results['general_stats']
    metadata = results['data']['metadata']
    
    with open(os.path.join(output_dir, 'general_summary.txt'), 'w') as f:
        f.write(f"===== {dataset_name.upper()} MASK ANALYSIS SUMMARY =====\n\n")
        f.write(f"Mask Size: {metadata.get('mask_size', 'Unknown')}\n")
        f.write(f"Embedding Dimension: {metadata.get('embedding_dim', 'Unknown')}\n")
        f.write(f"Validation Completeness: {metadata.get('val_completeness', 'Unknown')}\n")
        f.write(f"Validation Soundness: {metadata.get('val_soundness', 'Unknown')}\n\n")
        
        f.write("Training Set Statistics:\n")
        f.write(f"  Merlin Average Features: {general_stats['train_merlin_avg_features']:.2f} ± {general_stats['train_merlin_std_features']:.2f}\n")
        f.write(f"  Merlin Min/Max Features: {general_stats['train_merlin_min_features']:.0f}/{general_stats['train_merlin_max_features']:.0f}\n")
        f.write(f"  Morgana Average Features: {general_stats['train_morgana_avg_features']:.2f} ± {general_stats['train_morgana_std_features']:.2f}\n")
        f.write(f"  Morgana Min/Max Features: {general_stats['train_morgana_min_features']:.0f}/{general_stats['train_morgana_max_features']:.0f}\n")
        f.write(f"  Average Mask Overlap: {general_stats['train_mask_overlap_avg']:.2f} features ({general_stats['train_mask_overlap_pct']:.2f}%)\n\n")
        
        if 'val_merlin_avg_features' in general_stats:
            f.write("Validation Set Statistics:\n")
            f.write(f"  Merlin Average Features: {general_stats['val_merlin_avg_features']:.2f} ± {general_stats['val_merlin_std_features']:.2f}\n")
            f.write(f"  Merlin Min/Max Features: {general_stats['val_merlin_min_features']:.0f}/{general_stats['val_merlin_max_features']:.0f}\n")
            f.write(f"  Morgana Average Features: {general_stats['val_morgana_avg_features']:.2f} ± {general_stats['val_morgana_std_features']:.2f}\n")
            f.write(f"  Morgana Min/Max Features: {general_stats['val_morgana_min_features']:.0f}/{general_stats['val_morgana_max_features']:.0f}\n")
            f.write(f"  Average Mask Overlap: {general_stats['val_mask_overlap_avg']:.2f} features ({general_stats['val_mask_overlap_pct']:.2f}%)\n\n")
        
        f.write(f"Top {top_k} Most Frequent Merlin Features Overall:\n")
        for i, feat_idx in enumerate(general_stats['train_merlin_top_features'][:top_k]):
            freq = general_stats['train_merlin_feature_freqs'][feat_idx].item()
            f.write(f"  {i+1}. Feature {feat_idx}: {freq:.4f} ({freq*100:.2f}%)\n")
        
        f.write(f"\nTop {top_k} Most Frequent Morgana Features Overall:\n")
        for i, feat_idx in enumerate(general_stats['train_morgana_top_features'][:top_k]):
            freq = general_stats['train_morgana_feature_freqs'][feat_idx].item()
            f.write(f"  {i+1}. Feature {feat_idx}: {freq:.4f} ({freq*100:.2f}%)\n")
    
    # Per-class report
    class_stats = results['class_stats']
    
    with open(os.path.join(output_dir, 'per_class_summary.txt'), 'w') as f:
        f.write(f"===== {dataset_name.upper()} PER-CLASS MASK ANALYSIS =====\n\n")
        
        classes = sorted(class_stats['train'].keys())
        
        for cls in classes:
            cls_stats = class_stats['train'][cls]
            unique_features = class_stats['train_unique_features'][cls]
            
            f.write(f"Class {cls} (Count: {cls_stats['count']})\n")
            f.write(f"  Merlin Average Features: {cls_stats['merlin_avg_features']:.2f} ± {cls_stats['merlin_std_features']:.2f}\n")
            f.write(f"  Morgana Average Features: {cls_stats['morgana_avg_features']:.2f} ± {cls_stats['morgana_std_features']:.2f}\n")
            f.write(f"  Unique Features: {unique_features['unique_count']} ({unique_features['unique_pct']:.2f}%)\n")
            
            f.write(f"  Top {top_k} Merlin Features:\n")
            for i, (feat_idx, value) in enumerate(zip(cls_stats['merlin_top_features'], cls_stats['merlin_top_values'])):
                unique_marker = " *" if feat_idx in unique_features['unique_features'] else ""
                f.write(f"    {i+1}. Feature {feat_idx}: {value:.4f} ({value*100:.2f}%){unique_marker}\n")
            
            f.write(f"  Top {top_k} Morgana Features:\n")
            for i, (feat_idx, value) in enumerate(zip(cls_stats['morgana_top_features'], cls_stats['morgana_top_values'])):
                f.write(f"    {i+1}. Feature {feat_idx}: {value:.4f} ({value*100:.2f}%)\n")
            
            f.write("\n")

def cluster_and_analyze_masks(data: Dict, 
                            selector: str = 'merlin',
                            n_clusters: int = 50,
                            tolerance: int = 2):
    """
    Cluster masks and compute precision/entropy per cluster
    """
    from sklearn.cluster import KMeans
    from collections import Counter
    
    key = f"val_{selector}_masks"
    masks = data[key].cpu().numpy()  # (N, D)
    labels = data['val_labels'].cpu().numpy()  # (N,)
    
    print(f"Clustering {len(masks)} masks into {n_clusters} clusters...")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_assignments = kmeans.fit_predict(masks)
    centroids = kmeans.cluster_centers_
    
    print(f"Found {len(np.unique(cluster_assignments))} clusters")
    
    # Analyze each cluster
    cluster_results = {}
    
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_assignments == cluster_id)
        if cluster_mask.sum() == 0:
            continue
            
        # Get samples in this cluster
        cluster_labels = labels[cluster_mask]
        cluster_size = len(cluster_labels)
        
        # Compute class distribution
        class_counts = Counter(cluster_labels)
        total = sum(class_counts.values())
        class_probs = {cls: count/total for cls, count in class_counts.items()}
        
        # Compute precision (max class probability) and entropy
        max_prob = max(class_probs.values()) if class_probs else 0.0
        entropy = -sum(p * np.log(p + 1e-8) for p in class_probs.values())
        
        # Compute average Hamming distance to centroid
        cluster_masks_binary = masks[cluster_mask]
        centroid_binary = (centroids[cluster_id] > 0.5).astype(np.uint8)
        
        hamming_distances = []
        for mask in cluster_masks_binary:
            hamming_dist = np.sum(mask != centroid_binary)
            hamming_distances.append(hamming_dist)
        
        avg_hamming_dist = np.mean(hamming_distances)
        
        cluster_results[cluster_id] = {
            'size': cluster_size,
            'precision': max_prob,
            'entropy': entropy,
            'class_distribution': class_probs,
            'dominant_class': max(class_counts, key=class_counts.get) if class_counts else -1,
            'avg_hamming_to_centroid': avg_hamming_dist,
            'centroid': centroid_binary
        }
    
    return cluster_results, centroids

# Add to your main analysis function:
def main():
    args = parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load masks and data
    data = load_masks_and_data(args.masks_path, args.data_path, args.batch_size)
    
    # If requested, run only the precision & entropy computation and exit early
    if args.compute_precision_entropy and args.compute_only:
        print("Running precision & entropy computation only...")
        pe_results = compute_precision_and_entropy_masks(
            data,
            selector=args.selector,
            target_class=args.target_class,
            tolerance=args.tolerance,
            batch_size=max(256, args.batch_size),
            seed=args.seed,
            device='cuda'
        )
        out_csv = os.path.join(output_dir, f'precision_entropy_{args.selector}.csv')
        with open(out_csv, 'w') as fh:
            fh.write('class,average_precision,conditional_entropy,avg_occurrence\n')
            for cls, (ap, ent, avg_occurrence) in sorted(pe_results.items()):
                fh.write(f'{cls},{ap:.6f},{ent:.6f},{avg_occurrence:.2f}\n')
        print(f"Precision/entropy results saved to {out_csv}")
        return

    # Run analyses (original full pipeline)
    print("Analyzing general mask statistics...")
    general_stats = analyze_general_stats(data)
    
    print("Analyzing per-class statistics...")
    class_stats = analyze_per_class_stats(data, args.top_k)
    
    # Optional precision & entropy computation (when not in compute_only mode)
    if args.compute_precision_entropy:
        if args.cluster_analysis:
            print("Running cluster-based analysis...")
            cluster_results, centroids = cluster_and_analyze_masks(
                data, 
                selector=args.selector,
                n_clusters=args.n_clusters,
                tolerance=args.tolerance
            )
            
            # Save cluster results
            cluster_csv = os.path.join(output_dir, f'cluster_analysis_{args.selector}.csv')
            with open(cluster_csv, 'w') as f:
                f.write('cluster_id,size,precision,entropy,dominant_class,avg_hamming_dist\n')
                for cid, stats in cluster_results.items():
                    f.write(f'{cid},{stats["size"]},{stats["precision"]:.6f},'
                        f'{stats["entropy"]:.6f},{stats["dominant_class"]},'
                        f'{stats["avg_hamming_to_centroid"]:.2f}\n')

            print(f"Cluster analysis saved to {cluster_csv}")
        print("Computing average precision and conditional entropy (mask-based)...")
        pe_results = compute_precision_and_entropy_masks(
            data,
            selector=args.selector,
            target_class=args.target_class,
            tolerance=args.tolerance,
            batch_size=max(256, args.batch_size),
            seed=args.seed,
            device='cuda'
        )
        # save results to CSV
        out_csv = os.path.join(output_dir, f'precision_entropy_{args.selector}.csv')
        with open(out_csv, 'w') as fh:
            fh.write('class,average_precision,conditional_entropy,avg_occurrence\n')
            for cls, (ap, ent, avg_occurrence) in sorted(pe_results.items()):
                fh.write(f'{cls},{ap:.6f},{ent:.6f},{avg_occurrence:.2f}\n')
        print(f"Precision/entropy results saved to {out_csv}")
    
    # Compile results
    results = {
        'data': data,
        'general_stats': general_stats,
        'class_stats': class_stats
    }
    
    # Generate reports and visualizations
    print("Generating reports...")
    generate_reports(results, output_dir, args.dataset, args.top_k)
    
    print("Generating visualizations...")
    generate_visualizations(results, output_dir, args.dataset, args.plot_heatmap)
    
    # Save the most important features to a CSV for further analysis
    print("Saving feature importance data...")
    # Per-class top features
    per_class_features = []
    for cls, stats in class_stats['train'].items():
        for i, (feat_idx, val) in enumerate(zip(stats['merlin_top_features'], stats['merlin_top_values'])):
            is_unique = feat_idx in class_stats['train_unique_features'][cls]['unique_features']
            per_class_features.append({
                'class': cls,
                'rank': i+1,
                'feature_idx': feat_idx,
                'frequency': val,
                'percentage': val*100,
                'is_unique': is_unique
            })
    
    df_class_features = pd.DataFrame(per_class_features)
    df_class_features.to_csv(os.path.join(output_dir, 'per_class_top_features.csv'), index=False)
    
    # Global feature frequencies
    global_features = []
    for i in range(len(general_stats['train_merlin_feature_freqs'])):
        global_features.append({
            'feature_idx': i,
            'merlin_frequency': general_stats['train_merlin_feature_freqs'][i].item(),
            'morgana_frequency': general_stats['train_morgana_feature_freqs'][i].item(),
            'merlin_percentage': general_stats['train_merlin_feature_freqs'][i].item() * 100,
            'morgana_percentage': general_stats['train_morgana_feature_freqs'][i].item() * 100,
        })
    
    df_global_features = pd.DataFrame(global_features)
    df_global_features.to_csv(os.path.join(output_dir, 'global_feature_frequencies.csv'), index=False)
    
    print(f"Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()