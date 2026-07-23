from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainerConfig:
    epochs: int
    approach: str
    soundness_threshold: float
    debug: bool = False
    seed: int = 42
    wandb: bool = False
    early_stopping: bool = True
    patience: int = 10
    extract_masks: bool = False  # Whether to extract and save masks
    masks_output_path: Optional[str] = None  # Path to save the masks
    checkpoint_path: Optional[str] = None  # Path to specific checkpoint to load
    skip_training: bool = False  # Skip training and only extract masks
    # Analysis / compute flags for mask analysis
    compute_precision_entropy: bool = False
    compute_only: bool = False
    selector: str = "merlin"
    tolerance: int = 2
    target_class: Optional[int] = None
    analysis_seed: int = 42
    use_class_weights: bool = True

@dataclass
class DatasetConfig:
    root_dir: str
    dataset: str
    batch_size: int
    l1_penalty_splice: float = 0.10
    add_normalization: bool = False
    num_workers: int = 4
    vocab_size: int = 10000
    use_nonsparse: bool = False

@dataclass
class BooleanConfig:
    save_model: bool = False
    use_amp: bool = False
    cuda_benchmark: bool = False

@dataclass
class ModelConfig:
    model: str
    imagenet_pretrained: bool = False
    lr: float = 0.001
    weight_decay: float = 0.0
    hidden_dim: int = 128
    dropout: float = 0.0
    low_rank: int = 64
    pretrained_model: bool = False
    pretrained_path: Optional[str] = None
    load_sfw_model: bool = False
    
    # Learning rate scheduler options
    use_lr_scheduler: bool = False
    lr_scheduler_type: str = "plateau"  # Options: "plateau", "step", "cosine", "exponential"
    lr_scheduler_patience: int = 5      # For plateau scheduler
    lr_scheduler_factor: float = 0.1    # Factor to reduce LR by
    lr_scheduler_min_lr: float = 1e-6   # Minimum learning rate
    lr_scheduler_step_size: int = 10    # For step scheduler
    lr_scheduler_gamma: float = 0.1     # For step and exponential schedulers
    
@dataclass
class FeatureSelectorConfig:
    feature_selector_architecture: str
    segmentation_method: str = "topk"
    mask_size: int = 32
    lr_merlin: float = 0.05
    lr_morgana: float = 0.05
    gamma: float = 0.1
    l1_penalty_coefficient: Optional[float] = None
    overlap_weight: float = 0.0
    track_mask_statistics: bool = False
    feature_inclusion_weight: float = 0.0
    weight_decay_merlin: float = 0
    weight_decay_morgana: float = 0
    prioritize_nonzero: bool = False
    # arguments for mask reconstruction using SFW as ground truth
    use_saved_masks: bool = False
    saved_masks_path: Optional[str] = None
    mask_reconstruction_weight: float = 0.1
    use_continuous_reconstruction: bool = False
    load_sfw_model: bool = False
    weight_positive_examples: bool = False
