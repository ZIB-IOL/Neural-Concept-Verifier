import torch

def calculate_mask_overlap_statistics(inputs, binary_mask, threshold=1e-5):
    """
    Calculate overlap statistics between mask and non-zero input features.
    
    Args:
        inputs: Input tensor [batch_size x embedding_dim]
        binary_mask: Binary mask tensor [batch_size x embedding_dim]
        threshold: Threshold to consider input values as non-zero
        
    Returns:
        dict: Dictionary containing overlap statistics
    """
    batch_size = inputs.size(0)
    mask_overlap_pcts = []  # % of mask features that are non-zero in input
    feature_inclusion_pcts = []  # % of non-zero input features included in mask
    nonzero_counts = []
    
    for i in range(batch_size):
        # Get indices where input and mask are non-zero
        input_nonzeros = torch.nonzero(inputs[i].abs() > threshold, as_tuple=True)[0]
        mask_nonzeros = torch.nonzero(binary_mask[i], as_tuple=True)[0]
        
        # Calculate overlap statistics
        num_input_nonzeros = len(input_nonzeros)
        nonzero_counts.append(num_input_nonzeros)
        
        # Convert to sets and find intersection
        input_set = set(input_nonzeros.cpu().numpy())
        mask_set = set(mask_nonzeros.cpu().numpy())
        common = len(input_set.intersection(mask_set))
        
        # Calculate percentage of mask that overlaps with non-zero inputs
        if len(mask_set) > 0:
            mask_overlap_pct = (common / len(mask_set)) * 100
            mask_overlap_pcts.append(mask_overlap_pct)
        
        # Calculate percentage of non-zero features that are included in mask
        if len(input_set) > 0:
            feature_inclusion_pct = (common / len(input_set)) * 100
            feature_inclusion_pcts.append(feature_inclusion_pct)
    
    # Calculate average statistics
    avg_mask_overlap = sum(mask_overlap_pcts) / len(mask_overlap_pcts) if mask_overlap_pcts else 0
    avg_feature_inclusion = sum(feature_inclusion_pcts) / len(feature_inclusion_pcts) if feature_inclusion_pcts else 0
    avg_nonzeros = sum(nonzero_counts) / len(nonzero_counts) if nonzero_counts else 0
    
    return {
        'avg_overlap': avg_mask_overlap,  # For backward compatibility
        'avg_mask_overlap': avg_mask_overlap,  # % of mask overlapping with non-zero features
        'avg_feature_inclusion': avg_feature_inclusion,  # % of non-zero features included in mask
        'avg_nonzeros': avg_nonzeros,
        'nonzero_counts': nonzero_counts
    }


class IndexedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that returns (data, target, index)"""
    def __init__(self, dataset, subset_indices=None):
        self.dataset = dataset
        self.subset_indices = subset_indices
    
    def __getitem__(self, index):
        if self.subset_indices is not None:
            # Get the original dataset index if using a subset
            orig_index = self.subset_indices[index]
            data, target = self.dataset[orig_index]
            return data, target, orig_index
        else:
            data, target = self.dataset[index]
            return data, target, index
    
    def __len__(self):
        if self.subset_indices is not None:
            return len(self.subset_indices)
        else:
            return len(self.dataset)
