import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from typing import Dict
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from data.generate_COCOLogic import COCOLogicDataset, load_category_mapping, split_dataset


from torchvision import datasets, transforms
import torch.nn.functional as F
from copy import deepcopy

from models.classifier import LinearClassifier, MLPClassifier, NonLinearClassifier, LowRankLinearClassifier, ShallowMLPRelu, ShallowMLPGelu
from models.feature_selector_models import (
    MLPFeatureSelector, SetTransformerFeatureSelector, SimpleMLPFeatureSelector,
    NarrowMLPFeatureSelector, LightweightTransformerSelector, SmallLightweightTransformerSelector,
    GatedFeatureSelector, SparseAwareMLP, SparseFeatureAttention, DeepResidualMLPFeatureSelector, # Add the new models here
    LinearFeatureSelector, LowRankLinearFeatureSelector, ResidualFeatureSelector, BottleneckFeatureSelector,
    DenseNetFeatureSelector, GatedResidualNetwork, SimpleNet
)
from config.config_dataclass import TrainerConfig, DatasetConfig, ModelConfig, BooleanConfig, FeatureSelectorConfig
from merlin_arthur_framework.feature_selectors import SFWFeatureSelector, NeuralFeatureSelector, MorganaCriterion, PixelFeatureSelector

import sys
import h5py
import os
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
from utils.logger import get_accuracy, get_balanced_soundness_cocologic

from utils.mask_utils import calculate_mask_overlap_statistics, IndexedDataset


class BaseTrainer:
    def __init__(
        self, 
        trainer_config: TrainerConfig,
        dataset_config: DatasetConfig,
        model_config: ModelConfig, 
        bool_config: BooleanConfig,
        feature_selector_config: FeatureSelectorConfig,
        logger=None
    ):
        """Initialize trainer with complete configuration setup
        
        Args:
            trainer_config: Configuration for training parameters
            dataset_config: Configuration for dataset parameters
            model_config: Configuration for model parameters
            bool_config: Configuration for boolean parameters
            logger: Logger for logging training metrics
        """
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Setup all configurations
        self._setup_trainer_config(trainer_config)
        self._setup_dataset_config(dataset_config)
        self._setup_model_config(model_config)
        self._setup_bool_config(bool_config)
        self._setup_feature_selector_config(feature_selector_config)
        
        # Initialize empty attributes
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None

        # set seed
        self._setup_seed(self.seed)

        # Store logger instance
        self.logger = logger

    def _setup_trainer_config(self, config: TrainerConfig) -> None:
        """Setup trainer configuration parameters"""
        # Core training parameters
        self.epochs = config.epochs
        self.approach = config.approach
        self.soundness_threshold = config.soundness_threshold
        self.debug = config.debug
        self.seed = config.seed
        self.use_wandb = config.wandb
        self.early_stopping = config.early_stopping
        self.patience = config.patience
        self.trainer_config = config
        self.use_class_weights = config.use_class_weights
                
    def _setup_dataset_config(self, config: DatasetConfig) -> None:
        """Setup dataset configuration parameters"""
        self.root_dir = config.root_dir
        self.dataset_name = config.dataset
        self.l1_penalty_splice = config.l1_penalty_splice
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.vocab_size = config.vocab_size
        self.use_nonsparse = config.use_nonsparse
        self.dataset_config = config

        if self.dataset_name.upper() == 'CIFAR100' or self.dataset_name.upper() == 'PIXELCIFAR100':
            self.num_classes = 100
        elif self.dataset_name.upper() == 'IMAGENET' or self.dataset_name.upper() == 'PIXELIMAGENET':
            self.num_classes = 1000
        elif self.dataset_name.upper() in ['COCOLOGIC7', 'PIXELCOCOLOGIC7']:
            self.num_classes = 7
        elif self.dataset_name.upper() in ['COCOLOGIC8', 'PIXELCOCOLOGIC8']:
            self.num_classes = 8
        elif self.dataset_name.upper() in ['COCOLOGIC10', 'PIXELCOCOLOGIC10']:
            self.num_classes = 10
        else:
            # Add a default or raise an error for unknown datasets
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
    def _setup_model_config(self, config: ModelConfig) -> None:
        """Setup model configuration parameters"""
        self.model_name = config.model
        self.imagenet_pretrained = config.imagenet_pretrained
        self.learning_rate = config.lr
        self.weight_decay = config.weight_decay
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.low_rank = config.low_rank
        self.pretrained_model = config.pretrained_model
        self.pretrained_path = config.pretrained_path
        self.load_sfw_model = config.load_sfw_model
        
        # Learning rate scheduler parameters
        self.use_lr_scheduler = config.use_lr_scheduler
        self.lr_scheduler_type = config.lr_scheduler_type
        self.lr_scheduler_patience = config.lr_scheduler_patience
        self.lr_scheduler_factor = config.lr_scheduler_factor
        self.lr_scheduler_min_lr = config.lr_scheduler_min_lr
        self.lr_scheduler_step_size = config.lr_scheduler_step_size
        self.lr_scheduler_gamma = config.lr_scheduler_gamma
        
        self.model_config = config
        
    def _setup_bool_config(self, config: BooleanConfig) -> None:
        """Setup boolean configuration parameters"""
        self.save_model = config.save_model
        self.use_amp = config.use_amp
        self.cuda_benchmark = config.cuda_benchmark
        self.boolean_config = config
    
    def _setup_feature_selector_config(self, config: FeatureSelectorConfig) -> None:
        """Setup feature selector configuration parameters"""
        self.feature_selector_architecture = config.feature_selector_architecture
        self.segmentation_method = config.segmentation_method
        self.mask_size = config.mask_size
        self.lr_merlin = config.lr_merlin
        self.lr_morgana = config.lr_morgana
        self.gamma = config.gamma
        self.l1_penalty_coefficient = config.l1_penalty_coefficient
        self.overlap_weight = config.overlap_weight
        self.weight_decay_merlin = config.weight_decay_merlin
        self.weight_decay_morgana = config.weight_decay_morgana
        # Add the attributes for mask reconstruction regularization
        self.use_saved_masks = config.use_saved_masks
        self.saved_masks_path = config.saved_masks_path
        self.mask_reconstruction_weight = config.mask_reconstruction_weight
        self.use_continuous_reconstruction = config.use_continuous_reconstruction
        self.weight_positive_examples = config.weight_positive_examples
        self.load_sfw_model = config.load_sfw_model


        self.feature_selector_config = config

        # Only initialize SFW feature selectors here as they don't need embedding_dim
        if self.approach == "sfw":
            # Initialize feature selector (Merlin)
            self.merlin = SFWFeatureSelector(
                mask_size=self.mask_size,
                mode="merlin",
                lr_merlin=self.lr_merlin,
                l1_penalty_coefficient=self.l1_penalty_coefficient,
                overlap_weight=self.overlap_weight
            ).to(self.device)

            # Initialize feature selector (Morgana)
            self.morgana = SFWFeatureSelector(
                mask_size=self.mask_size,
                mode="morgana",
                lr_merlin=self.lr_morgana,
                l1_penalty_coefficient=self.l1_penalty_coefficient,
                overlap_weight=self.overlap_weight,
                idk_class=self.num_classes
            ).to(self.device)
        
        # The neural feature selector initialization will be deferred until setup_model() is called

    def _setup_seed(self, seed: int) -> None:
        """Setup all seeds for full reproducibility
        
        Args:
            seed: Integer seed value
        """        
        # Python
        random.seed(seed)
        # Numpy
        np.random.seed(seed)
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        # Deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"\nSeed set to {seed} for reproducibility!")

    def setup_data(self):
        """Initialize dataloaders based on dataset configuration"""
        if self.dataset_name.upper() in ['CIFAR100', 'IMAGENET', 'COCOLOGIC7', 'COCOLOGIC8', 'COCOLOGIC10']:
            # Get embeddings path based on dataset and L1 penalty
            dataset_folder = f"embeddings_{self.dataset_name.lower()}"
            file_prefix = self.dataset_name.lower()
            
            # Build paths in a standardized way
            embeddings_dir = os.path.join(self.root_dir, dataset_folder)
            
            # Add vocabulary size to path structure
            vocab_dir = os.path.join(embeddings_dir, f"vocab_{self.vocab_size}")
            
            # Construct path based on sparse/nonsparse flag
            if self.use_nonsparse:
                # For nonsparse embeddings, no L1 penalty needed
                embeddings_dir = os.path.join(vocab_dir, "nonsparse")
            else:
                # For sparse embeddings with L1 penalty
                embeddings_dir = os.path.join(vocab_dir, f'l1_{self.l1_penalty_splice:.3f}')
            
            embeddings_path = os.path.join(embeddings_dir, f'{file_prefix}_splice_embeddings_full.h5')
            
            print("\nChecking dataset paths:")
            print(f"Dataset: {self.dataset_name}")
            print(f"Root directory: {self.root_dir}")
            print(f"Embedding vocabulary size: {self.vocab_size}")
            print(f"Embedding type: {'nonsparse' if self.use_nonsparse else 'sparse'}")
            if not self.use_nonsparse:
                print(f"L1 penalty: {self.l1_penalty_splice:.3f}")
            print(f"Full embeddings path: {embeddings_path}")
            
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(
                    f"Embeddings file not found at {embeddings_path}.\n"
                    "Please run precompute_embeddings.py first with:\n"
                    f"python src/data/precompute_embeddings.py "
                    f"--dataset_root YOUR_ARG1 "
                    f"--dataset {self.dataset_name.lower()} "
                    f"--save_dir YOUR_ARG2 "
                    f"--l1_penalty_splice YOUR_ARG3"
                )

            print(f"Loading {self.dataset_name} SpLiCE embeddings from {embeddings_path}")
            
            # Load embeddings and labels from HDF5 file
            train_dataset, test_dataset, embedding_dim = self._load_splice_embeddings(embeddings_path)

        elif self.dataset_name.upper() in ['PIXELCIFAR100', 'PIXELIMAGENET']:
            # Load splice for preprocessing
            try:
                # Completely clear splice from sys.modules to force a fresh import
                for mod_name in list(sys.modules.keys()):
                    if mod_name == 'splice' or mod_name.startswith('splice.'):
                        del sys.modules[mod_name]
                
                # Temporarily remove our custom paths
                saved_paths = []
                for path in list(sys.path):
                    if 'splice_customized' in path or 'MerlinArthur-SpLiCE' in path:
                        sys.path.remove(path)
                        saved_paths.append(path)
                        
                # Try to locate and import system splice directly
                import site
                sys_site_packages = site.getsitepackages()
                for site_path in sys_site_packages:
                    if site_path not in sys.path:
                        sys.path.insert(0, site_path)
                        
                # Now import GitHub splice
                import importlib
                splice = importlib.import_module('splice')
                print(f"Using GitHub SpLiCE from: {splice.__file__}")
                is_sparse = True
            
            except ImportError:
                print("Could not import GitHub SpLiCE. Make sure it's installed with pip.")

            # Load images 
            preprocess_transform = splice.get_preprocess("open_clip:ViT-B-32")
            if self.dataset_name.lower() == 'pixelcifar100':
                train_dataset = datasets.CIFAR100(root=self.root_dir, train=True, download=False, transform=preprocess_transform)
                test_dataset = datasets.CIFAR100(root=self.root_dir, train=False, download=False, transform=preprocess_transform)
            # For PIXELIMAGENET case
            if self.dataset_name.lower() == 'pixelimagenet':
                # Use the CORRECT ResNet preprocessing
                preprocess_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                train_dir = os.path.join(self.root_dir, 'imagenet', 'train')
                val_dir = os.path.join(self.root_dir, 'imagenet', 'val')
                
                print(f"Loading ImageNet from {os.path.join(self.root_dir, 'imagenet')}")
                train_dataset = datasets.ImageFolder(train_dir, transform=preprocess_transform)
                test_dataset = datasets.ImageFolder(val_dir, transform=preprocess_transform)

                if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
                    raise FileNotFoundError(
                        f"ImageNet directory not found at {os.path.join(self.root_dir, 'imagenet')}.\n"
                        "Please download ImageNet manually and organize it with train/ and val/ subdirectories."
                    )
                
                print(f"Loading ImageNet from {os.path.join(self.root_dir, 'imagenet')}")
                train_dataset = datasets.ImageFolder(train_dir, transform=preprocess_transform)
                test_dataset = datasets.ImageFolder(val_dir, transform=preprocess_transform)
                
        elif self.dataset_name.upper() in ['PIXELCOCOLOGIC7', 'PIXELCOCOLOGIC8', 'PIXELCOCOLOGIC10']:
            # Extract version number from dataset name
            version = int(self.dataset_name.upper().replace('PIXELCOCOLOGIC', ''))
            
            # Set up image preprocessing transforms
            preprocess_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # COCO dataset paths - adapt these to your setup
            train_annotation_file = os.path.join(self.root_dir, 'annotations', 'instances_train2017.json')
            val_annotation_file = os.path.join(self.root_dir, 'annotations', 'instances_val2017.json')
            train_image_dir = os.path.join(self.root_dir, 'train2017')
            val_image_dir = os.path.join(self.root_dir, 'val2017')


            # Load category mappings
            category_id_to_name = load_category_mapping(train_annotation_file)
            
            print(f"Loading PixelCOCOLogic{version} from {os.path.join(self.root_dir, 'coco')}")
            
            # Create datasets
            train_dataset = COCOLogicDataset(
                annotation_file=train_annotation_file,
                image_dir=train_image_dir,
                category_id_to_name=category_id_to_name,
                transform=preprocess_transform,
                filter_no_labels=True,
                exclusive_label=True,
                exclusive_match_only=True,
                log_statistics=True,
                version=version
            )
            
            test_dataset = COCOLogicDataset(
                annotation_file=val_annotation_file,
                image_dir=val_image_dir,
                category_id_to_name=category_id_to_name,
                transform=preprocess_transform,
                filter_no_labels=True,
                exclusive_label=True,
                exclusive_match_only=True,
                log_statistics=True,
                version=version
            )

            # Check if COCO directories exist
            if not (os.path.exists(train_image_dir) and os.path.exists(val_image_dir)):
                raise FileNotFoundError(
                    f"COCO directory not found at {os.path.join(self.root_dir, 'coco')}.\n"
                    "Please download COCO dataset and organize it with train2017/, val2017/, and annotations/ subdirectories."
                )
            
            print(f"PixelCOCOLogic{version} loaded successfully")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(test_dataset)}")


        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented yet")

        # If in debug mode, use only a small subset of the data
        if self.debug:
            train_dataset, test_dataset = self._create_debug_subset(train_dataset, test_dataset)

        # Create data loaders
        self.train_loader, self.val_loader = self._create_data_loaders(train_dataset, test_dataset)
        
        # Print dataset info
        if self.dataset_name.upper() in ['CIFAR100', 'IMAGENET', 'COCOLOGIC10']:
            self._print_dataset_info(train_dataset, test_dataset, embedding_dim)

    def _load_splice_embeddings(self, embeddings_path):
        """Helper method to load SpLiCE embeddings from HDF5 file"""
        with h5py.File(embeddings_path, 'r') as f:
            train_embeddings = torch.from_numpy(f['train_embeddings'][:])
            train_labels = torch.from_numpy(f['train_labels'][:])
            test_embeddings = torch.from_numpy(f['test_embeddings'][:])
            test_labels = torch.from_numpy(f['test_labels'][:])
        
        train_dataset = TensorDataset(train_embeddings, train_labels)
        test_dataset = TensorDataset(test_embeddings, test_labels)
        
        # Store these values for later use
        self.embedding_dim = train_embeddings.shape[1]
        self.num_classes = len(torch.unique(train_labels))
        
        return train_dataset, test_dataset, self.embedding_dim

    def _create_debug_subset(self, train_dataset, test_dataset):
        """Helper method to create debug subsets of data"""
        train_subset_indices = list(range(2 * self.batch_size))
        test_subset_indices = list(range(2 * self.batch_size))
        train_dataset = Subset(train_dataset, train_subset_indices)
        test_dataset = Subset(test_dataset, test_subset_indices)
        return train_dataset, test_dataset

    def _create_data_loaders(self, train_dataset, test_dataset):
        """Helper method to create data loaders"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        return train_loader, val_loader

    def _print_dataset_info(self, train_dataset, test_dataset, embedding_dim=None):
        """Helper method to print dataset information"""
        print(f"\nDataset {self.dataset_name} initialized:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(test_dataset)}")
        if embedding_dim is not None:
            print(f"Embedding dimension: {embedding_dim}")
        
    def setup_model(self):
        """Initialize model, optimizer and criterion with robust model loading"""
        
        # First check if we need to load a pre-trained model
        if self.pretrained_model and self.approach != "regular":
            # Determine the checkpoint path
            if self.pretrained_path:
                # Use explicitly provided path
                checkpoint_path = None
                
                # Try to find the exact model first
                exact_path = os.path.join(
                    self.pretrained_path,
                    f'{self.dataset_name}',
                    f'vocab_{self.vocab_size}',
                    f"{'nonsparse' if self.use_nonsparse else f'l1_{self.l1_penalty_splice:.3f}'}",
                    f"{self.model_name}" if self.model_name else f'splice_l1_{self.l1_penalty_splice:.3f}_{self.model_name}', 
                    'best_model.pth')
                
                # Check for SFW model path with mask size parameter
                sfw_path = os.path.join(
                    self.pretrained_path,
                    f'{self.dataset_name}',
                    f'vocab_{self.vocab_size}',
                    f"{'nonsparse' if self.use_nonsparse else f'l1_{self.l1_penalty_splice:.3f}'}",
                    f"{self.model_name}_mask_{self.mask_size}",
                    'best_model.pth'
                )                
                # First check if we are specifically asked to load an SFW model
                if hasattr(self, 'load_sfw_model') and self.load_sfw_model:
                    if os.path.exists(sfw_path):
                        checkpoint_path = sfw_path
                        print(f"Loading SFW model as specified by load_sfw_model flag")
                    else:
                        print(f"Warning: Requested SFW model not found at {sfw_path}")
                        # Fall back to regular model
                        if os.path.exists(exact_path):
                            checkpoint_path = exact_path
                else:
                    # Try regular path first, then SFW path as fallback
                    if os.path.exists(exact_path):
                        checkpoint_path = exact_path
                    elif os.path.exists(sfw_path):
                        checkpoint_path = sfw_path
                        print(f"Note: Using SFW model checkpoint as regular checkpoint was not found")
            else:
                # Construct path based on local configuration
                if self.approach == "sfw":
                    checkpoint_dir = os.path.join(
                        'src', 
                        'checkpoints',
                        f'{self.dataset_name}', 
                        f'splice_l1_{self.l1_penalty_splice:.3f}_{self.model_name}'
                    )
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                else:
                    # For non-SFW approaches, try both regular and SFW model paths
                    regular_dir = os.path.join(
                        'src', 
                        'checkpoints',
                        f'{self.dataset_name}', 
                        f'splice_l1_{self.l1_penalty_splice:.3f}_{self.model_name}'
                    )
                    regular_path = os.path.join(regular_dir, 'best_model.pth')
                    
                    sfw_dir = os.path.join(
                        'src', 
                        'checkpoints',
                        f'{self.dataset_name}', 
                        f'sfw_splice_l1_{self.l1_penalty_splice:.3f}_mask_{self.mask_size}'
                    )
                    sfw_path = os.path.join(sfw_dir, 'best_model.pth')
                    
                    if hasattr(self, 'load_sfw_model') and self.load_sfw_model:
                        checkpoint_path = sfw_path if os.path.exists(sfw_path) else regular_path
                    else:
                        checkpoint_path = regular_path if os.path.exists(regular_path) else sfw_path
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"\nLoading pretrained model from {checkpoint_path}")
                
                # Load checkpoint to determine architecture
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' not in checkpoint:
                    raise ValueError(f"Invalid checkpoint format: 'model_state_dict' not found in {checkpoint_path}")
                
                state_dict = checkpoint['model_state_dict']
                
                # Detect model architecture from state_dict
                model_type = self._detect_model_architecture(state_dict)
                print(f"Detected model architecture: {model_type}")
                
                if self.model_name == "resnet" or self.model_name == "resnet50":
                    # init the same resnet model
                    if self.model_name == "resnet":
                        from torchvision.models import resnet18, ResNet18_Weights
                        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                    elif self.model_name == "resnet50":
                        from torchvision.models import resnet50, ResNet50_Weights
                        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, self.num_classes+1)
                    self.model = model

                    # Fix the prefix issue by removing 'resnet.' from all keys
                    new_state_dict = {}
                    for key in state_dict:
                        if key.startswith('resnet.'):
                            new_key = key[7:]  # Remove the 'resnet.' prefix
                            new_state_dict[new_key] = state_dict[key]
                        else:
                            new_state_dict[key] = state_dict[key]
                    
                    # Use this modified state_dict instead
                    state_dict = new_state_dict  
                else:
                    # Initialize model based on detected architecture
                    if model_type == "linear":
                        # Use linear.weight instead of fc.weight
                        input_dim = state_dict['linear.weight'].shape[1]
                        output_dim = state_dict['linear.weight'].shape[0]
                        print(f"Input dim: {input_dim}, Output dim: {output_dim}")
                        self.model = LinearClassifier(input_dim=input_dim, num_classes=output_dim)
                        
                    elif model_type == "lowrank":
                        # Use linear.weight and rank_linear.weight instead of fc.weight and u_key
                        input_dim = state_dict['linear.weight'].shape[1]
                        output_dim = state_dict['rank_linear.weight'].shape[0]
                        rank = state_dict['linear.weight'].shape[0]  # Assuming this is the rank
                        print(f"Input dim: {input_dim}, Output dim: {output_dim}, Rank: {rank}")
                        self.model = LowRankLinearClassifier(input_dim=input_dim, num_classes=output_dim, rank=rank)
                        
                    elif model_type == "mlp":
                        input_dim = state_dict['network.0.weight'].shape[1]
                        hidden_dim = state_dict['network.0.weight'].shape[0]
                        output_dim = state_dict['network.6.weight'].shape[0]
                        print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
                        self.model = MLPClassifier(input_dim=input_dim, num_classes=output_dim, hidden_dim=hidden_dim, dropout=self.dropout)
                        
                    elif model_type == "nonlinear":
                        input_dim = state_dict['network.0.weight'].shape[1]
                        hidden_dim = state_dict['network.0.weight'].shape[0]
                        output_dim = state_dict['network.4.weight'].shape[0]
                        print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
                        self.model = NonLinearClassifier(input_dim=input_dim, num_classes=output_dim, hidden_dim=hidden_dim, dropout=self.dropout)
                        
                    elif model_type == "shallowmlprelu":
                        input_dim = state_dict['network.0.weight'].shape[1]
                        hidden_dim = state_dict['network.0.weight'].shape[0]
                        output_dim = state_dict['network.4.weight'].shape[0]
                        print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
                        self.model = ShallowMLPRelu(input_dim=input_dim, num_classes=output_dim, hidden_dim=hidden_dim, dropout=self.dropout)

                    elif model_type == "shallowmlpgelu":
                        input_dim = state_dict['network.0.weight'].shape[1]
                        hidden_dim = state_dict['network.0.weight'].shape[0]
                        output_dim = state_dict['network.4.weight'].shape[0]
                        print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
                        self.model = ShallowMLPGelu(input_dim=input_dim, num_classes=output_dim, hidden_dim=hidden_dim, dropout=self.dropout)
                    else:
                        raise ValueError(f"Unknown model architecture detected in {checkpoint_path}")
                
                # Move model to device
                self.model = self.model.to(self.device)
                
                # Load state dict
                self.model.load_state_dict(state_dict)
                
                # Store the embedding dim from the model for feature selector setup
                first_weight_layer = next(k for k in state_dict.keys() if 'weight' in k and len(state_dict[k].shape) > 1)
                self.embedding_dim = state_dict[first_weight_layer].shape[1]
                
                # Print information about the loaded model
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
                print(f"Validation accuracy: {checkpoint.get('best_val_acc', checkpoint.get('val_completeness', 'N/A'))}")
                
            else:
                print(f"\nWARNING: Pretrained model not found at {checkpoint_path}")
                print("Initializing model with provided architecture parameters...")
                self._initialize_model_from_params()
        else:
            # No pretrained model, initialize with parameters
            self._initialize_model_from_params()
        
        # Set up neural feature selectors if using nn approach
        if self.approach == "nn":
            self._setup_neural_feature_selectors()
            
            # Load saved masks if using reconstruction regularization
            if self.feature_selector_config.use_saved_masks:
                self._load_saved_masks()
                print(f"Loaded saved masks from {self.feature_selector_config.saved_masks_path}")

        elif self.approach == "unet":
            self.merlin_model = SimpleNet(
                n_channels = 3, 
                bilinear = True, 
                apply_sigmoid = True
                )
            
            # Create a deep copy for morgana
            self.morgana_model = deepcopy(self.merlin_model)
            
            self.merlin = PixelFeatureSelector(
                mask_size=self.mask_size,
                mode="merlin",
                idk_class=self.num_classes,
                model=self.merlin_model
            ).to(self.device)

            self.morgana = PixelFeatureSelector(
                mask_size=self.mask_size,
                mode="morgana",
                idk_class=self.num_classes,
                model=self.morgana_model
            ).to(self.device)

            self.merlin_optimizer = torch.optim.Adam(self.merlin.parameters(), lr=self.lr_merlin, weight_decay=self.weight_decay_merlin)
            self.morgana_optimizer = torch.optim.Adam(self.morgana.parameters(), lr=self.lr_morgana, weight_decay=self.weight_decay_morgana)
        
        # Setup optimizer and criterion
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Optionally compute class weights
        if self.trainer_config.use_class_weights:
            print("Training with class weights...")
            y_train_all = [label.item() for _, label in self.train_loader.dataset]
            
            # Get the actual unique classes present in the training data
            unique_classes = np.unique(y_train_all)
            print(f"Classes present in training data: {sorted(unique_classes)}")
            
            # Only compute weights for classes that actually exist
            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=unique_classes,  # Only use actual classes in data
                y=y_train_all
            )
            
            # Create weight tensor with zeros for missing classes
            num_model_classes = self.num_classes + 1  # Model outputs 11 classes
            full_class_weights = torch.ones(num_model_classes, dtype=torch.float)
            
            # Assign computed weights only to classes that exist in data
            for i, class_id in enumerate(unique_classes):
                full_class_weights[class_id] = class_weights[i]
            
            # The reject class (class 10) keeps weight=1.0 since it's not in training data
            full_class_weights = full_class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=full_class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
                    
        # Initialize learning rate scheduler if requested
        self.scheduler = None
        if self.use_lr_scheduler:
            if self.lr_scheduler_type == "plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',  # We're monitoring loss, so mode is min
                    factor=self.lr_scheduler_factor,
                    patience=self.lr_scheduler_patience,
                    verbose=True,
                    min_lr=self.lr_scheduler_min_lr
                )
                print(f"Using ReduceLROnPlateau scheduler with patience={self.lr_scheduler_patience}, factor={self.lr_scheduler_factor}")
            
            elif self.lr_scheduler_type == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=self.lr_scheduler_step_size,
                    gamma=self.lr_scheduler_gamma,
                    verbose=True
                )
                print(f"Using StepLR scheduler with step_size={self.lr_scheduler_step_size}, gamma={self.lr_scheduler_gamma}")
            
            elif self.lr_scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epochs,  # Full cycle length
                    eta_min=self.lr_scheduler_min_lr,
                    verbose=True
                )
                print(f"Using CosineAnnealingLR scheduler with T_max={self.epochs}, eta_min={self.lr_scheduler_min_lr}")
            
            elif self.lr_scheduler_type == "exponential":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=self.lr_scheduler_gamma,
                    verbose=True
                )
                print(f"Using ExponentialLR scheduler with gamma={self.lr_scheduler_gamma}")
        
        # Print model info
        self._print_model_info()
        
        # Print feature selector info if using SFW or NN approach
        if self.approach in ["sfw", "nn"]:
            self._print_feature_selector_info()

    def _detect_model_architecture(self, state_dict):
        """Detect model architecture from state dict"""
        keys = list(state_dict.keys())
        print(f"Debug - Model keys: {keys}")  # Add debug output
        
        # Linear model check 
        if 'linear.weight' in state_dict and 'linear.bias' in state_dict and 'rank_linear.weight' not in state_dict and len(keys) == 2:
            return "linear"
        
        # LowRank model check
        if 'linear.weight' in state_dict and 'rank_linear.weight' in state_dict:
            return "lowrank"
        
        # MLP and NonLinear checks
        if 'network.0.weight' in state_dict:
            # NonLinear has 5 layers (check for network.4.weight)
            if any('network.4.weight' in k for k in keys):
                return "nonlinear"
            # MLP has 7 layers (check for network.6.weight)
            elif any('network.6.weight' in k for k in keys):
                return "mlp"
        
        # Alternative checks for more complex architectures
        if any('fc1.weight' in k for k in keys) and any('fc3.weight' in k for k in keys):
            return "nonlinear"
        elif any('fc1.weight' in k for k in keys) and any('fc2.weight' in k for k in keys):
            return "mlp"
        elif any('fc.weight' in k for k in keys) and any('u_' in k for k in keys):
            return "lowrank"
        elif any('fc.weight' in k for k in keys):
            return "linear"
        
        # Include the full key list in the error message for better debugging
        raise ValueError(f"Could not determine model architecture from keys: {keys}")

    def _initialize_model_from_params(self):
        """Initialize model using configuration parameters"""
        if self.model_name.lower() == 'linear':
            self.model = LinearClassifier(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes + 1,  # + (1 * self.approach != "regular"),  # +1 for the unknown class
            )
        elif self.model_name.lower() == 'lowrank':
            self.model = LowRankLinearClassifier(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes + 1,  # + (1 * self.approach != "regular"),  # +1 for the unknown class
                rank=self.low_rank
            )
        elif self.model_name.lower() == 'mlp':
            self.model = MLPClassifier(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes + 1, # + (1 * self.approach != "regular"),  # +1 for the unknown class
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
        elif self.model_name.lower() == 'nonlinear':
            self.model = NonLinearClassifier(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes + 1, # + (1 * self.approach != "regular"),  # +1 for the unknown class
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
        elif self.model_name.lower() == 'shallowmlprelu':
            self.model = ShallowMLPRelu(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes + 1, # + (1 * self.approach != "regular"),  # +1 for the unknown class
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
        elif self.model_name.lower() == 'shallowmlpgelu':
            self.model = ShallowMLPRelu(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes + 1, # + (1 * self.approach != "regular"),  # +1 for the unknown class
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
        elif self.model_name.lower() == 'shallowmlprelu':
            self.model = ShallowMLPRelu(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes+1,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
        elif self.model_name.lower() == 'shallowmlpgelu':
            self.model = ShallowMLPGelu(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes+1,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
        elif self.model_name.lower() == 'resnet':
            # Import the standard model and weights
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            
            # Replace final fully connected layer
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes+1)
            
            self.model = model
        elif self.model_name.lower() == 'resnet50':
            # Import the standard model and weights
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            
            # Replace final fully connected layer
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes+1)
            
            self.model = model
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented yet")
        
        # Move model to device
        self.model = self.model.to(self.device)

    def _setup_neural_feature_selectors(self):
        """Initialize neural feature selectors after embedding_dim is known"""
        # Initialize neural feature selector models
        if self.feature_selector_architecture == "mlp":
            self.merlin_model = MLPFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
            self.morgana_model = MLPFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
        elif self.feature_selector_architecture == "settransformer":
            self.merlin_model = SetTransformerFeatureSelector(
                input_dim=self.embedding_dim,
                num_slots=self.mask_size,
                num_blocks=1,
                dim_hidden=128,
                num_heads=4,
                ln=False,
                dropout=0
            )
            self.morgana_model = SetTransformerFeatureSelector(
                input_dim=self.embedding_dim,
                num_slots=self.mask_size,
                num_blocks=1,
                dim_hidden=128,
                num_heads=4,
                ln=False,
                dropout=0
            )
        elif self.feature_selector_architecture == "simplemlpfeatureselector":
            self.merlin_model = SimpleMLPFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=32,
                dropout=0
            )
            self.morgana_model = SimpleMLPFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=32,
                dropout=0
            )
        elif self.feature_selector_architecture == "narrowmlpfeatureselector":
            self.merlin_model = NarrowMLPFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=32,
                dropout=0
            )
            self.morgana_model = NarrowMLPFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=32,
                dropout=self.dropout
            )
        elif self.feature_selector_architecture == "lightweighttransformer":
            self.merlin_model = LightweightTransformerSelector(
                input_dim=self.embedding_dim,
                num_heads=4,
                dim_feedforward=128,
                dropout=0
            )
            self.morgana_model = LightweightTransformerSelector(
                input_dim=self.embedding_dim,
                num_heads=4,
                dim_feedforward=128,
                dropout=0
            )
        elif self.feature_selector_architecture == "smalllightweighttransformer":
            self.merlin_model = SmallLightweightTransformerSelector(
                input_dim=self.embedding_dim,
                num_heads=4,
                dim_feedforward=128,
                dropout=0
            )
            self.morgana_model = SmallLightweightTransformerSelector(
                input_dim=self.embedding_dim,
                num_heads=4,
                dim_feedforward=128,
                dropout=0
            )
        elif self.feature_selector_architecture == "gatedfeatureselector":
            self.merlin_model = GatedFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=128,
                dropout=0
            )
            self.morgana_model = GatedFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=128,
                dropout=0
            )
        elif self.feature_selector_architecture == "sparseawaremlp":
            self.merlin_model = SparseAwareMLP(
                input_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                dropout=0
            )
            self.morgana_model = SparseAwareMLP(
                input_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                dropout=0
            )
        elif self.feature_selector_architecture == "sparsefeatureattention":
            self.merlin_model = SparseFeatureAttention(
                input_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                dropout=0
            )
            self.morgana_model = SparseFeatureAttention(
                input_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                dropout=0
            )
        elif self.feature_selector_architecture == "deepresidualmlpfeatureselector":
            self.merlin_model = DeepResidualMLPFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=128,
                dropout=0
            )
            self.morgana_model = DeepResidualMLPFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=128,
                dropout=0
            )
        # New models added below
        elif self.feature_selector_architecture == "linearfeatureselector":
            self.merlin_model = LinearFeatureSelector(
                input_dim=self.embedding_dim
            )
            self.morgana_model = LinearFeatureSelector(
                input_dim=self.embedding_dim
            )
        elif self.feature_selector_architecture == "lowranklinearfeatureselector":
            # Default rank of 64, but can use self.low_rank if provided
            rank = getattr(self, 'low_rank', 64)
            self.merlin_model = LowRankLinearFeatureSelector(
                input_dim=self.embedding_dim, 
                rank=rank
            )
            self.morgana_model = LowRankLinearFeatureSelector(
                input_dim=self.embedding_dim, 
                rank=rank
            )
        elif self.feature_selector_architecture == "residualfeatureselector":
            self.merlin_model = ResidualFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=128,
                dropout=0.1
            )
            self.morgana_model = ResidualFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=128,
                dropout=0.1
            )
        elif self.feature_selector_architecture == "bottleneckfeatureselector":
            self.merlin_model = BottleneckFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=256,
                bottleneck_dim=32,
                dropout=0.1
            )
            self.morgana_model = BottleneckFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=256,
                bottleneck_dim=32,
                dropout=0.1
            )
        elif self.feature_selector_architecture == "densenetfeatureselector":
            self.merlin_model = DenseNetFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=64,
                num_layers=4,
                growth_rate=16,
                dropout=0.1
            )
            self.morgana_model = DenseNetFeatureSelector(
                input_dim=self.embedding_dim,
                hidden_dim=64,
                num_layers=4,
                growth_rate=16,
                dropout=0.1
            )
        elif self.feature_selector_architecture == "gatedresidualnetwork":
            self.merlin_model = GatedResidualNetwork(
                input_dim=self.embedding_dim,
                hidden_dim=128,
                num_layers=3,
                dropout=0.1
            )
            self.morgana_model = GatedResidualNetwork(
                input_dim=self.embedding_dim,
                hidden_dim=128,
                num_layers=3,
                dropout=0.1
            )
        else:
            raise NotImplementedError(f"Feature selector architecture {self.feature_selector_architecture} not implemented yet")

        # Initialize feature selector wrappers
        self.merlin = NeuralFeatureSelector(
            mask_size=self.mask_size,
            lr=self.lr_merlin,
            model=self.merlin_model,
            mode="merlin",
            idk_class=self.num_classes,
            prioritize_nonzero=self.feature_selector_config.prioritize_nonzero
        ).to(self.device)
        
        self.morgana = NeuralFeatureSelector(
            mask_size=self.mask_size,
            lr=self.lr_morgana,
            model=self.morgana_model,
            mode="morgana",
            idk_class=self.num_classes,
            prioritize_nonzero=self.feature_selector_config.prioritize_nonzero
        ).to(self.device)

        # Initialize optimizers
        self.merlin_optimizer = torch.optim.Adam(
            self.merlin.model.parameters(), 
            lr=self.lr_merlin, 
            weight_decay=self.weight_decay_merlin
        )
        
        self.morgana_optimizer = torch.optim.Adam(
            self.morgana.model.parameters(), 
            lr=self.lr_morgana, 
            weight_decay=self.weight_decay_morgana
        )

    def _print_model_info(self):
        """Helper method to print model information"""
        print("\nModel setup:")
        print(f"Architecture: {self.model_name}")
        print(f"Pretrained: {self.pretrained_model}")
        num_params = sum(p.numel() for p in self.model.parameters()) # type: ignore
        print(f"Number of parameters: {num_params:,}")  # Formatted with commas
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {self.device}")

        # print rest of the model info
        if self.model_name.lower() == 'mlp' or self.model_name.lower() == 'nonlinear':
            print(f"Hidden dimension: {self.hidden_dim}")
            print(f"Dropout: {self.dropout}")
        elif self.model_name.lower() == 'lowrank':
            print(f"Rank: {self.low_rank}")


    def _print_feature_selector_info(self):
        """Helper method to print feature selector information"""
        if self.approach == "sfw":
            print("\nFeature Selector setup:")
            print(f"Method: Stochastic Frank-Wolfe (SFW)")
            print(f"Mask size (sparsity): {self.mask_size} features")
            print(f"L1 penalty coefficient: {self.l1_penalty_coefficient}")
            print(f"Overlap weight: {self.overlap_weight}")
            print(f"Merlin learning rate: {self.lr_merlin}")
            print(f"Morgana learning rate: {self.lr_morgana}")
            print(f"Gamma (Morgana weight): {self.gamma}")
            print(f"Device: {self.device}")
            
            # Calculate percentage of features kept
            sparsity_percentage = (self.mask_size / self.embedding_dim) * 100
            print(f"Sparsity ratio: {sparsity_percentage:.2f}% of features kept")
        
        elif self.approach == "nn":
            print("\nNeural Feature Selector setup:")
            print(f"Method: Neural Network")
            print(f"Architecture: {self.feature_selector_architecture}")

            num_params_merlin = sum(p.numel() for p in self.merlin_model.parameters()) # type: ignore
            num_params_morgana = sum(p.numel() for p in self.morgana_model.parameters()) # type: ignore
            print(f"Merlin Feature Selector parameters: {num_params_merlin:,}")
            print(f"Morgana Feature Selector parameters: {num_params_morgana:,}")

            print(f"Feature Selector hidden dimension: {self.hidden_dim}")
            print(f"Feature Selector dropout: {self.dropout}")
            print(f"Mask size (sparsity): {self.mask_size} features")
            print(f"L1 penalty coefficient: {self.l1_penalty_coefficient}")
            print(f"Merlin learning rate: {self.lr_merlin}")
            print(f"Merlin weight decay: {self.weight_decay_merlin}")
            print(f"Morgana learning rate: {self.lr_morgana}")
            print(f"Morgana weight decay: {self.weight_decay_morgana}")
            print(f"Gamma (Morgana weight): {self.gamma}")
            print(f"Device: {self.device}")
            
            # Calculate percentage of features kept
            sparsity_percentage = (self.mask_size / self.embedding_dim) * 100
            print(f"Sparsity ratio: {sparsity_percentage:.2f}% of features kept")

    def _create_train_indices_mapping(self):
        """Create a mapping from sample indices to their positions in the saved masks"""
        print("Creating index mapping for reconstruction loss...")
        
        # Convert the dataset to IndexedDataset that returns indices with data
        if isinstance(self.train_loader.dataset, torch.utils.data.Subset):
            # Handle Subset case (used in debug mode)
            original_dataset = self.train_loader.dataset.dataset
            indices = self.train_loader.dataset.indices
            indexed_dataset = IndexedDataset(original_dataset, indices)
        else:
            # Regular dataset
            indexed_dataset = IndexedDataset(self.train_loader.dataset)
        
        # Recreate the DataLoader with the indexed dataset
        self.train_loader = DataLoader(
            indexed_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print("Index mapping created successfully")

    def _create_val_indices_mapping(self):
        """Create a mapping from sample indices to their positions in the saved masks"""
        print("Creating validation index mapping for reconstruction loss evaluation...")
        
        # Convert the dataset to IndexedDataset that returns indices with data
        if isinstance(self.val_loader.dataset, torch.utils.data.Subset):
            # Handle Subset case (used in debug mode)
            original_dataset = self.val_loader.dataset.dataset
            indices = self.val_loader.dataset.indices
            indexed_dataset = IndexedDataset(original_dataset, indices)
        else:
            # Regular dataset
            indexed_dataset = IndexedDataset(self.val_loader.dataset)
        
        # Recreate the DataLoader with the indexed dataset
        # Important: keep shuffle=False for validation loader
        self.val_loader = DataLoader(
            indexed_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Keep validation set ordered
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print("Validation index mapping created successfully")

    def train(self):
        """Run the complete training loop based on specified approach
        
        Returns:
            dict: Dictionary containing best validation metrics
        """
        if self.approach == "regular":
            return self._train_regular()
        elif self.approach == "sfw":
            return self._train_sfw()
        elif self.approach == "nn":
            return self._train_nn()
        elif self.approach == "posthoc":
            raise NotImplementedError("Post-hoc approach not implemented yet")
        elif self.approach == 'unet':
            return self._train_unet()
        else:
            raise ValueError(f"Unknown approach: {self.approach}")

    def _train_regular(self):
        """Regular training approach
        
        Returns:
            dict: Dictionary containing best validation metrics
        """
        print(f"\nStarting regular training for {self.epochs} epochs...")
        best_metrics = {
            'best_val_acc': 0,
            'best_epoch': -1,
            'val_loss': float('inf')
        }

        # Simple tracking for true best metrics
        best_accuracy = 0
        best_balanced_accuracy = 0
        best_acc_epoch = -1
        best_balanced_acc_epoch = -1

        no_improvement = 0  # Counter for early stopping

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            # Log metrics to wandb if enabled
            if self.logger is not None:
                self.logger.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['train_loss'],
                    'train/accuracy': train_metrics['train_acc'],
                    'train/balanced_accuracy': train_metrics['train_balanced_acc'],
                    'val/loss': val_metrics['val_loss'],
                    'val/accuracy': val_metrics['val_acc'],
                    'val/balanced_accuracy': val_metrics['val_balanced_acc'],
                    'learning_rate': self.learning_rate,
                }, step=epoch)
            
            # Print metrics
            print(f"\nTraining Loss: {train_metrics['train_loss']:.4f}")
            print(f"Training Accuracy: {train_metrics['train_acc']:.2f}%")
            print(f"Validation Loss: {val_metrics['val_loss']:.4f}")
            print(f"Validation Accuracy: {val_metrics['val_acc']:.2f}%")
            
            # Early stopping based on validation loss improvement
            if val_metrics['val_loss'] < best_metrics['val_loss']:
                best_metrics.update({
                    'best_val_acc': val_metrics['val_acc'],
                    'best_val_balanced_acc': val_metrics['val_balanced_acc'],
                    'best_epoch': epoch,
                    'val_loss': val_metrics['val_loss'],
                    'train_acc': train_metrics['train_acc'],
                    'train_balanced_acc': train_metrics['train_balanced_acc'],  # ADD THIS LINE
                    'train_loss': train_metrics['train_loss']
                })
                no_improvement = 0  # Reset counter on improvement

                if self.save_model:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        **best_metrics
                    }
                    
                    # Determine checkpoint directory based on pretrained_path if available
                    if self.pretrained_path:
                        # Use the provided pretrained_path
                        checkpoint_dir = os.path.join(
                            self.pretrained_path,
                            f'{self.dataset_name}',
                            f'vocab_{self.vocab_size}',
                            f"{'nonsparse' if self.use_nonsparse else f'l1_{self.l1_penalty_splice:.3f}'}",
                            f'{self.model_name}'
                        )
                    else:
                        # Use the default path in src/checkpoints
                        checkpoint_dir = os.path.join(
                            'src', 
                            'checkpoints',
                            f'{self.dataset_name}',
                            f'vocab_{self.vocab_size}',
                            f"{'nonsparse' if self.use_nonsparse else f'l1_{self.l1_penalty_splice:.3f}'}",
                            f'{self.model_name}'
                        )
                    
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    
                    self.save_checkpoint(checkpoint, checkpoint_path)
                    print(f"Updated best model checkpoint at {checkpoint_path}")
                    print(f"Validation accuracy: {best_metrics['best_val_acc']:.2f}%")
            else:
                no_improvement += 1
                print(f"No improvement for {no_improvement} epoch(s).")
            

            # ADD: Simple best accuracy tracking
            if val_metrics['val_acc'] > best_accuracy:
                best_accuracy = val_metrics['val_acc']
                best_acc_epoch = epoch
                
            if val_metrics['val_balanced_acc'] > best_balanced_accuracy:
                best_balanced_accuracy = val_metrics['val_balanced_acc']
                best_balanced_acc_epoch = epoch


            # Check early stopping criterion
            if self.early_stopping and no_improvement >= self.patience:
                print(f"Early stopping triggered after {no_improvement} epochs with no improvement.")
                break
                    
        print(f"\nTraining completed! Best validation accuracy: {best_metrics['best_val_acc']:.2f}% "
              f"at epoch {best_metrics['best_epoch']+1}")
        
        # Log best metrics to wandb for sweep comparison
        if self.logger is not None:
            self.logger.log({
                'best/val_accuracy': best_metrics['best_val_acc'],
                'best/val_balanced_accuracy': best_metrics['best_val_balanced_acc'],
                'best/val_loss': best_metrics['val_loss'],
                'best/train_accuracy': best_metrics['train_acc'],
                'best/train_balanced_accuracy': best_metrics['train_balanced_acc'],
                'best/train_loss': best_metrics['train_loss'],
                'best/epoch': best_metrics['best_epoch']+1,
                # True best metrics over all epochs
                'best_overall/val_accuracy': best_accuracy,
                'best_overall/val_balanced_accuracy': best_balanced_accuracy,
                'best_overall/acc_epoch': best_acc_epoch+1,
                'best_overall/balanced_acc_epoch': best_balanced_acc_epoch+1
            })

        return best_metrics

    def _train_sfw(self):
        """Merlin-Arthur training approach using SFW
        
        Returns:
            dict: Dictionary containing best validation metrics
        """
        
        print(f"\nStarting Merlin-Arthur training with SFW for {self.epochs} epochs...")
                
        best_metrics = {
            'best_combined_metric': 0,
            'best_epoch': -1,
            'val_loss': float('inf'),
            'val_completeness': 0,
            'val_soundness': 0,
            'train_completeness': 0,
            'train_soundness': 0,
            'train_loss': float('inf')
        }
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train and validate
            train_metrics = self._train_epoch_sfw()
            val_metrics = self._validate_sfw()
            
            # Apply learning rate scheduler step if enabled
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                if self.lr_scheduler_type == "plateau":
                    self.scheduler.step(val_metrics['val_loss'])  # Pass validation loss for plateau
                else:
                    self.scheduler.step()  # For other schedulers
                
                # Log current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current Arthur learning rate: {current_lr:.6f}")
                if self.logger is not None:
                    self.logger.log({'lr/arthur': current_lr}, step=epoch)
            
            # Log metrics to wandb if enabled
            if self.logger is not None:
                metrics_dict = {
                    'epoch': epoch,
                    'train/loss': train_metrics['train_loss'],
                    'train/completeness': train_metrics['train_completeness'],
                    'train/soundness': train_metrics['train_soundness'],
                    'val/loss': val_metrics['val_loss'],
                    'val/completeness': val_metrics['val_completeness'],
                    'val/soundness': val_metrics['val_soundness'],
                    'mask/sparsity': self.mask_size,
                }
                
                # Include learning rate in metrics if scheduler is enabled
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    metrics_dict['lr/arthur'] = self.optimizer.param_groups[0]['lr']
                
                # Only include mask statistics if tracking is enabled
                if self.feature_selector_config.track_mask_statistics:
                    metrics_dict.update({
                        'mask/merlin_overlap': train_metrics.get('merlin_feature_overlap', 0),
                        'mask/merlin_inclusion': train_metrics.get('merlin_feature_inclusion', 0),
                        'mask/morgana_overlap': train_metrics.get('morgana_feature_overlap', 0),
                        'mask/morgana_inclusion': train_metrics.get('morgana_feature_inclusion', 0),
                        'inputs/nonzero_features': train_metrics.get('nonzero_features', 0)
                    })
                
                self.logger.log(metrics_dict, step=epoch)

            # Consolidated Epoch Summary
            print("\n" + "=" * 60)
            print("EPOCH SUMMARY")
            print("=" * 60)
            print(f"Epoch {epoch+1}/{self.epochs}")
            print("-" * 60)
            print("Training Statistics:")
            print(f"   Loss            : {train_metrics['train_loss']:.4f}")
            print(f"   Completeness    : {train_metrics['train_completeness']:.2f}%")
            print(f"   Soundness       : {train_metrics['train_soundness']:.2f}%")
            print("-" * 60)
            print("-" * 60)
            print("Validation Statistics:")
            print(f"   Loss            : {val_metrics['val_loss']:.4f}")
            print(f"   Completeness    : {val_metrics['val_completeness']:.2f}%")
            print(f"   Soundness       : {val_metrics['val_soundness']:.2f}%")
            print("-" * 60)

            # Add this section for the learning rate
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                print(f"   Learning Rate    : {self.optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 60)

            if self.feature_selector_config.track_mask_statistics:
                print("Mask Statistics (on Train Dataset):")
                print(f"   Avg non-zero features    : {train_metrics['nonzero_features']:.1f}")
                print(f"   Merlin mask overlap      : {train_metrics['merlin_feature_overlap']:.1f}%")
                print(f"   Merlin feature inclusion : {train_metrics['merlin_feature_inclusion']:.1f}%")
                print(f"   Morgana mask overlap     : {train_metrics['morgana_feature_overlap']:.1f}%")
                print(f"   Morgana feature inclusion: {train_metrics['morgana_feature_inclusion']:.1f}%")
            print("=" * 60)

            # Update best metrics and save model
            if val_metrics['val_soundness'] > self.soundness_threshold:
                # If soundness threshold is met, add a large bonus to ensure it's better than sub-threshold models
                # but still maintains the comp+sound ordering among models above 75%
                combined_metric = val_metrics['val_completeness'] + val_metrics['val_soundness'] + 100
            else:
                combined_metric = val_metrics['val_completeness'] + val_metrics['val_soundness']
                
            if combined_metric > best_metrics['best_combined_metric']:
                best_metrics.update({
                    'best_combined_metric': combined_metric,
                    'best_epoch': epoch,
                    'val_loss': val_metrics['val_loss'],
                    'val_completeness': val_metrics['val_completeness'],
                    'val_soundness': val_metrics['val_soundness'],
                    'train_completeness': train_metrics['train_completeness'],
                    'train_soundness': train_metrics['train_soundness'],
                    'train_loss': train_metrics['train_loss']
                })
                
                if self.save_model:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'merlin_state_dict': self.merlin.state_dict(),
                        'morgana_state_dict': self.morgana.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_merlin': self.lr_merlin,
                        'lr_morgana': self.lr_morgana,
                        **best_metrics
                    }                    
                    # Determine checkpoint directory based on pretrained_path if available
                    if self.pretrained_path:
                        # Use the provided pretrained_path
                        checkpoint_dir = os.path.join(
                            self.pretrained_path,
                            f'{self.dataset_name}',
                            f'vocab_{self.vocab_size}',
                            f"{'nonsparse' if self.use_nonsparse else f'l1_{self.l1_penalty_splice:.3f}'}",
                            f"{self.model_name}_mask_{self.mask_size}"  # Include model name before mask size
                        )                    
                    else:
                        # Use the default path in src/checkpoints
                        checkpoint_dir = os.path.join(
                            'src', 
                            'checkpoints', 
                            f'{self.dataset_name}',
                            f'vocab_{self.vocab_size}',
                            f"{'nonsparse' if self.use_nonsparse else f'sfw_splice_l1_{self.l1_penalty_splice:.3f}'}_mask_{self.mask_size}"
                        )
                    
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    
                    self.save_checkpoint(checkpoint, checkpoint_path)
                    print(f"Updated best model checkpoint at {checkpoint_path}")
                    print(f"Combined metric: {best_metrics['best_combined_metric']:.2f}")
        
        # Print final summary with best metrics
        print(f"\nTraining completed!")
        print(f"Best combined metric at epoch {best_metrics['best_epoch']+1}")
        print(f"Best validation completeness: {best_metrics['val_completeness']:.2f}%")
        print(f"Best validation soundness: {best_metrics['val_soundness']:.2f}%")
        
        # Log best metrics to wandb for sweep comparison
        if self.logger is not None:
            self.logger.log({
                'best/combined_metric': best_metrics['best_combined_metric'],
                'best/val_completeness': best_metrics['val_completeness'],
                'best/val_soundness': best_metrics['val_soundness'],
                'best/val_loss': best_metrics['val_loss'],
                'best/train_completeness': best_metrics['train_completeness'],
                'best/train_soundness': best_metrics['train_soundness'],
                'best/train_loss': best_metrics['train_loss'],
                'best/epoch': best_metrics['best_epoch']+1
            })
        
        return best_metrics

    def _train_nn(self):
        """Merlin-Arthur training approach with neural network feature selectors
        
        Returns:
            dict: Dictionary containing best validation metrics
        """
        print(f"\nStarting Merlin-Arthur training with neural feature selectors for {self.epochs} epochs...")
                
        best_metrics = {
            'best_combined_metric': 0,
            'best_epoch': -1,
            'val_loss': float('inf'),
            'val_completeness': 0,
            'val_soundness': 0,
            'train_completeness': 0,
            'train_soundness': 0,
            'train_loss': float('inf'),
            'train_balanced_completeness': 0,
            'val_balanced_completeness': 0
        }

        # Simple tracking for true best completeness/soundness
        best_completeness = 0
        best_soundness = 0
        best_comp_epoch = -1
        best_sound_epoch = -1

        # ADD: For CoCoLogic datasets
        if 'cocologic' in self.dataset_name.lower():
            best_balanced_completeness = 0
            best_balanced_comp_epoch = -1
            best_balanced_soundness = 0
            best_balanced_sound_epoch = -1
        
        no_improvement = 0  # Counter for early stopping

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train and validate
            train_metrics = self._train_epoch_nn()
            val_metrics = self._validate_nn()
            
            # Apply learning rate scheduler step if enabled
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                if self.lr_scheduler_type == "plateau":
                    self.scheduler.step(val_metrics['val_loss'])  # Pass validation loss for plateau
                else:
                    self.scheduler.step()  # For other schedulers
                
                # Log current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current Arthur learning rate: {current_lr:.6f}")
                if self.logger is not None:
                    self.logger.log({'lr/arthur': current_lr}, step=epoch)

    
            # Log metrics to wandb if enabled
            if self.logger is not None:
                metrics_dict = {
                    'epoch': epoch,
                    'train/loss': train_metrics['train_loss'],
                    'train/completeness': train_metrics['train_completeness'],
                    'train/soundness': train_metrics['train_soundness'],
                    'val/loss': val_metrics['val_loss'],
                    'val/completeness': val_metrics['val_completeness'],
                    'val/soundness': val_metrics['val_soundness'],
                    'mask/sparsity': self.mask_size,
                }

                # Add balanced metrics only for CoCoLogic datasets
                if 'cocologic' in self.dataset_name.lower():
                    if 'train_balanced_completeness' in train_metrics:
                        metrics_dict['train/balanced_completeness'] = train_metrics['train_balanced_completeness']
                    if 'val_balanced_completeness' in val_metrics:
                        metrics_dict['val/balanced_completeness'] = val_metrics['val_balanced_completeness']
                    if 'train_balanced_soundness' in train_metrics:
                        metrics_dict['train/balanced_soundness'] = train_metrics['train_balanced_soundness']
                    if 'val_balanced_soundness' in val_metrics:
                        metrics_dict['val/balanced_soundness'] = val_metrics['val_balanced_soundness']

                self.logger.log(metrics_dict, step=epoch)

            if self.logger is not None and self.use_saved_masks:
                metrics_dict.update({
                    'val/merlin_recon_loss': val_metrics.get('val_merlin_recon_loss', 0),
                    'val/morgana_recon_loss': val_metrics.get('val_morgana_recon_loss', 0)
                })
                
                # Only include mask statistics if tracking is enabled
                if self.feature_selector_config.track_mask_statistics:
                    metrics_dict.update({
                        'mask/merlin_overlap': train_metrics.get('merlin_feature_overlap', 0),
                        'mask/merlin_inclusion': train_metrics.get('merlin_feature_inclusion', 0),
                        'mask/morgana_overlap': train_metrics.get('morgana_feature_overlap', 0),
                        'mask/morgana_inclusion': train_metrics.get('morgana_feature_inclusion', 0),
                        'inputs/nonzero_features': train_metrics.get('nonzero_features', 0)
                    })
                
                self.logger.log(metrics_dict, step=epoch)

            # Consolidated Epoch Summary
            print("\n" + "=" * 60)
            print("EPOCH SUMMARY")
            print("=" * 60)
            print(f"Epoch {epoch+1}/{self.epochs}")
            print("-" * 60)
            print("Training Statistics:")
            print(f"   Loss            : {train_metrics['train_loss']:.4f}")
            print(f"   Completeness    : {train_metrics['train_completeness']:.2f}%")
            if 'cocologic' in self.dataset_name.lower() and 'train_balanced_completeness' in train_metrics:
                print(f"   Balanced Comp.  : {train_metrics['train_balanced_completeness']:.2f}%")
            print(f"   Soundness       : {train_metrics['train_soundness']:.2f}%")
            print("-" * 60)
            print("Validation Statistics:")
            print(f"   Loss            : {val_metrics['val_loss']:.4f}")
            print(f"   Completeness    : {val_metrics['val_completeness']:.2f}%")
            if 'cocologic' in self.dataset_name.lower() and 'val_balanced_completeness' in val_metrics:
                print(f"   Balanced Comp.  : {val_metrics['val_balanced_completeness']:.2f}%")
            if 'cocologic' in self.dataset_name.lower() and 'val_balanced_soundness' in val_metrics:
                print(f"   Balanced Sound. : {val_metrics['val_balanced_soundness']:.2f}%")
            print(f"   Soundness       : {val_metrics['val_soundness']:.2f}%")
            print("-" * 60)

            # Add this section for the learning rate
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                print(f"   Learning Rate    : {self.optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 60)

            if self.feature_selector_config.track_mask_statistics:
                print("Mask Statistics (on Train Dataset):")
                print(f"   Avg non-zero features    : {train_metrics['nonzero_features']:.1f}")
                print(f"   Merlin mask overlap      : {train_metrics['merlin_feature_overlap']:.1f}%")
                print(f"   Merlin feature inclusion : {train_metrics['merlin_feature_inclusion']:.1f}%")
                print(f"   Morgana mask overlap     : {train_metrics['morgana_feature_overlap']:.1f}%")
                print(f"   Morgana feature inclusion: {train_metrics['morgana_feature_inclusion']:.1f}%")
            print("=" * 60)

            # Update best metrics and save model
            if val_metrics['val_soundness'] > self.soundness_threshold:
                combined_metric = val_metrics['val_completeness'] + val_metrics['val_soundness'] + 100
            else:
                combined_metric = val_metrics['val_completeness'] + val_metrics['val_soundness']
                
            if combined_metric > best_metrics['best_combined_metric']:
                best_metrics.update({
                    'best_combined_metric': combined_metric,
                    'best_epoch': epoch,
                    'val_loss': val_metrics['val_loss'],
                    'val_completeness': val_metrics['val_completeness'],
                    'val_soundness': val_metrics['val_soundness'],
                    'train_completeness': train_metrics['train_completeness'],
                    'train_soundness': train_metrics['train_soundness'],
                    'train_loss': train_metrics['train_loss']
                })
                
                if 'cocologic' in self.dataset_name.lower():
                    if 'train_balanced_completeness' in train_metrics:
                        best_metrics['train_balanced_completeness'] = train_metrics['train_balanced_completeness']
                    if 'val_balanced_completeness' in val_metrics:
                        best_metrics['val_balanced_completeness'] = val_metrics['val_balanced_completeness']
                        
                no_improvement = 0  # Reset counter on improvement
                
                if self.save_model:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'merlin_state_dict': self.merlin.state_dict(),
                        'morgana_state_dict': self.morgana.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'merlin_optimizer_state_dict': self.merlin_optimizer.state_dict(),
                        'morgana_optimizer_state_dict': self.morgana_optimizer.state_dict(),
                        **best_metrics
                    }
                    
                    # Determine checkpoint directory based on pretrained_path if available
                    if self.pretrained_path:
                        # Use the provided pretrained_path
                        checkpoint_dir = os.path.join(
                            self.pretrained_path,
                            f'{self.dataset_name}',
                            f'vocab_{self.vocab_size}',
                            f"{'nonsparse' if self.use_nonsparse else f'l1_{self.l1_penalty_splice:.3f}'}",
                            f"{self.model_name}_nn_mask_{self.mask_size}"  # Include model name before nn_mask
                        )
                    else:
                        # Use the default path in src/checkpoints
                        checkpoint_dir = os.path.join(
                            'src', 
                            'checkpoints', 
                            f'{self.dataset_name}',
                            f'vocab_{self.vocab_size}',
                            f"{'nonsparse_nn' if self.use_nonsparse else f'nn_splice_l1_{self.l1_penalty_splice:.3f}'}_mask_{self.mask_size}"
                        )
                    
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    
                    self.save_checkpoint(checkpoint, checkpoint_path)
                    print(f"Updated best model checkpoint at {checkpoint_path}")
                    print(f"Combined metric: {best_metrics['best_combined_metric']:.2f}")
            else:
                no_improvement += 1
                print(f"No improvement for {no_improvement} epoch(s).")
            

            # ADD: Simple best tracking AFTER the existing best metrics logic
            if val_metrics['val_completeness'] > best_completeness:
                best_completeness = val_metrics['val_completeness']
                best_comp_epoch = epoch
                
            if val_metrics['val_soundness'] > best_soundness:
                best_soundness = val_metrics['val_soundness']
                best_sound_epoch = epoch
            
            if 'cocologic' in self.dataset_name.lower() and 'val_balanced_completeness' in val_metrics:
                if val_metrics['val_balanced_completeness'] > best_balanced_completeness:
                    best_balanced_completeness = val_metrics['val_balanced_completeness']
                    best_balanced_comp_epoch = epoch

            if 'cocologic' in self.dataset_name.lower() and 'val_balanced_soundness' in val_metrics:
                if val_metrics['val_balanced_soundness'] > best_balanced_soundness:
                    best_balanced_soundness = val_metrics['val_balanced_soundness']
                    best_balanced_sound_epoch = epoch

            # Check early stopping criterion
            if self.early_stopping and no_improvement >= self.patience:
                print(f"Early stopping triggered after {no_improvement} epochs with no improvement.")
                break
        
        # Enhanced final summary (replace existing print statements)
        print(f"\nTraining completed!")
        print(f"Best combined metric at epoch {best_metrics['best_epoch']+1}")
        print(f"Best validation completeness (combined): {best_metrics['val_completeness']:.2f}%")
        print(f"Best validation completeness (overall): {best_completeness:.2f}% at epoch {best_comp_epoch+1}")
        print(f"Best validation soundness (combined): {best_metrics['val_soundness']:.2f}%")
        print(f"Best validation soundness (overall): {best_soundness:.2f}% at epoch {best_sound_epoch+1}")
        if 'cocologic' in self.dataset_name.lower():
            print(f"Best balanced completeness (overall): {best_balanced_completeness:.2f}% at epoch {best_balanced_comp_epoch+1}")
            print(f"Best balanced soundness (overall): {best_balanced_soundness:.2f}% at epoch {best_balanced_sound_epoch+1}")
        
        # Enhanced wandb logging (replace existing final_metrics logging)
        if self.logger is not None:
            final_metrics = {
                # Combined metric based (existing)
                'best/combined_metric': best_metrics['best_combined_metric'],
                'best/val_completeness': best_metrics['val_completeness'],
                'best/val_soundness': best_metrics['val_soundness'],
                'best/val_loss': best_metrics['val_loss'],
                'best/train_completeness': best_metrics['train_completeness'],
                'best/train_soundness': best_metrics['train_soundness'],
                'best/train_loss': best_metrics['train_loss'],
                'best/epoch': best_metrics['best_epoch']+1,
                
                # ADD: True best metrics
                'best_overall/val_completeness': best_completeness,
                'best_overall/val_soundness': best_soundness,
                'best_overall/comp_epoch': best_comp_epoch+1,
                'best_overall/sound_epoch': best_sound_epoch+1
            }
            
            # ADD balanced completeness for CoCoLogic:
            if 'cocologic' in self.dataset_name.lower():
                if 'train_balanced_completeness' in best_metrics:
                    final_metrics['best/train_balanced_completeness'] = best_metrics['train_balanced_completeness']
                if 'val_balanced_completeness' in best_metrics:
                    final_metrics['best/val_balanced_completeness'] = best_metrics['val_balanced_completeness']
                # ADD: True best balanced completeness
                final_metrics['best_overall/val_balanced_completeness'] = best_balanced_completeness
                final_metrics['best_overall/balanced_comp_epoch'] = best_balanced_comp_epoch+1
                final_metrics['best_overall/val_balanced_soundness'] = best_balanced_soundness
                final_metrics['best_overall/balanced_sound_epoch'] = best_balanced_sound_epoch+1

            
            self.logger.log(final_metrics)
        
        return best_metrics

    def train_epoch(self):
        """Train model for one epoch using current approach
        
        Returns:
            dict: Dictionary containing training metrics
        """
        if self.approach == "regular":
            return self._train_epoch_regular()
        elif self.approach == "sfw":
            return self._train_epoch_sfw()
        elif self.approach == "posthoc":
            raise NotImplementedError("Post-hoc training epoch not implemented yet")
        else:
            raise ValueError(f"Unknown approach: {self.approach}")

    def _train_epoch_regular(self):
        """Regular training for one epoch
        
        Returns:
            dict: Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        # Progress bar for training
        pbar = tqdm(self.train_loader, desc='Training')

        for inputs, targets in pbar:
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            y_true.extend(targets.cpu().numpy().tolist())

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            y_pred.extend(predicted.cpu().numpy().tolist())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/len(self.train_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Compute epoch metrics
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': 100. * correct / total,
            'train_balanced_acc': 100. * balanced_accuracy_score(y_true, y_pred)

        }
        
        return metrics

    def _train_epoch_sfw(self):
        """Train model for one epoch using SFW Merlin-Arthur approach"""
        self.model.train()
        total_loss = 0
        total_completeness = 0
        total_soundness = 0
        batch_count = 0
        
        # Track overlap statistics (only initialize if needed)
        merlin_batch_overlaps = []
        morgana_batch_overlaps = []
        merlin_feature_inclusions = []
        morgana_feature_inclusions = []
        batch_nonzeros = []
        
        # Progress bar for training
        pbar = tqdm(self.train_loader, desc='Training with Merlin-Arthur')
        
        for inputs, targets in pbar:
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Step 1: Optimize masks using SFW
            continuous_mask_merlin = self.merlin(inputs, targets, self.model)
            continuous_mask_morgana = self.morgana(inputs, targets, self.model)
            
            # Step 2: Convert to binary masks using top-k selection
            binary_mask_merlin = self.merlin.get_binary_mask(continuous_mask_merlin)
            binary_mask_morgana = self.morgana.get_binary_mask(continuous_mask_morgana)
            
            # Calculate overlap statistics ONLY if tracking is enabled
            if self.feature_selector_config.track_mask_statistics:
                merlin_stats = calculate_mask_overlap_statistics(inputs, binary_mask_merlin)
                morgana_stats = calculate_mask_overlap_statistics(inputs, binary_mask_morgana)
                
                # Store statistics for epoch averaging
                merlin_batch_overlaps.append(merlin_stats['avg_mask_overlap'])
                morgana_batch_overlaps.append(morgana_stats['avg_mask_overlap'])
                merlin_feature_inclusions.append(merlin_stats['avg_feature_inclusion'])
                morgana_feature_inclusions.append(morgana_stats['avg_feature_inclusion'])
                batch_nonzeros.extend(merlin_stats['nonzero_counts'])
            
            # Step 3: Apply mask and compute logits
            self.optimizer.zero_grad()
            masked_inputs_merlin = self.merlin.apply_mask(inputs, binary_mask_merlin)
            masked_inputs_morgana = self.morgana.apply_mask(inputs, binary_mask_morgana)
            logits_merlin = self.model(masked_inputs_merlin)
            logits_morgana = self.model(masked_inputs_morgana)

            # Step 4: Calculate loss and update model
            merlin_loss = self.merlin.criterion(logits_merlin, targets)
            morgana_loss = self.morgana.criterion(logits_morgana, targets)
            loss = merlin_loss + self.gamma * morgana_loss            
            
            # Backward pass for classifier
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()

            # Calculate accuracies (completeness and soundness)
            batch_completeness = get_accuracy(logits_merlin, targets, mode="merlin", idk_class=self.num_classes)
            batch_soundness = get_accuracy(logits_morgana, targets, mode="morgana", idk_class=self.num_classes)
            
            # Accumulate for epoch average
            total_completeness += batch_completeness
            total_soundness += batch_soundness
            batch_count += 1
                        
            # Update progress bar with appropriate info
            if self.feature_selector_config.track_mask_statistics:
                pbar.set_postfix({
                    'loss': f'{total_loss/batch_count:.4f}',
                    'comp': f'{100.*batch_completeness:.2f}%',
                    'sound': f'{100.*batch_soundness:.2f}%',
                    'M-ovlp': f'{merlin_stats["avg_mask_overlap"]:.1f}%',
                    'M-incl': f'{merlin_stats["avg_feature_inclusion"]:.1f}%'
                })
            else:
                pbar.set_postfix({
                    'loss': f'{total_loss/batch_count:.4f}',
                    'comp': f'{100.*batch_completeness:.2f}%',
                    'sound': f'{100.*batch_soundness:.2f}%'
                })
        
        # Calculate epoch-level statistics
        avg_merlin_overlap = np.mean(merlin_batch_overlaps) if merlin_batch_overlaps else 0
        avg_morgana_overlap = np.mean(morgana_batch_overlaps) if morgana_batch_overlaps else 0
        avg_merlin_inclusion = np.mean(merlin_feature_inclusions) if merlin_feature_inclusions else 0
        avg_morgana_inclusion = np.mean(morgana_feature_inclusions) if morgana_feature_inclusions else 0
        avg_nonzeros = np.mean(batch_nonzeros) if batch_nonzeros else 0
        avg_completeness = total_completeness / batch_count if batch_count > 0 else 0
        avg_soundness = total_soundness / batch_count if batch_count > 0 else 0
                
        # Compute epoch metrics
        metrics = {
            'train_loss': total_loss / batch_count,
            'train_completeness': 100. * avg_completeness,
            'train_soundness': 100. * avg_soundness,
            'merlin_feature_overlap': avg_merlin_overlap,
            'morgana_feature_overlap': avg_morgana_overlap,
            'merlin_feature_inclusion': avg_merlin_inclusion,
            'morgana_feature_inclusion': avg_morgana_inclusion,
            'nonzero_features': avg_nonzeros
        }
        
        return metrics

    def _train_epoch_nn(self):
        """Train model for one epoch using neural network feature selectors"""
        self.model.train()
        total_loss = 0
        total_completeness = 0
        total_soundness = 0
        total_merlin_recon_loss = 0
        total_morgana_recon_loss = 0
        batch_count = 0
        
        # Track statistics
        merlin_batch_overlaps = []
        morgana_batch_overlaps = []
        merlin_feature_inclusions = []
        morgana_feature_inclusions = []
        batch_nonzeros = []

        if 'cocologic' in self.dataset_name.lower():
            y_true_merlin = []
            y_pred_merlin = []
            y_true_morgana = []
            y_pred_morgana = []


        
        # Use reconstruction regularization if enabled (use the class variable)
        use_reconstruction = self.use_saved_masks and hasattr(self, 'saved_train_merlin_masks')
        recon_weight = self.mask_reconstruction_weight
        
        pbar = tqdm(self.train_loader, desc='Training with Neural Feature Selectors')
        
        for batch_idx, batch_data in enumerate(pbar):
            # Handle both regular data tuples and indexed data
            if use_reconstruction and len(batch_data) == 3:  # (inputs, targets, indices)
                inputs, targets, indices = batch_data
                # indices = indices.to(self.device)
            else:
                inputs, targets = batch_data
                indices = None
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Load saved masks for this batch if reconstruction is enabled
            if use_reconstruction and indices is not None:
                # Get masks corresponding to the specific sample indices
                saved_merlin_masks = self.saved_train_merlin_masks[indices].to(self.device)
                saved_morgana_masks = self.saved_train_morgana_masks[indices].to(self.device)
            else:
                saved_merlin_masks = None
                saved_morgana_masks = None
            
            # ===== MERLIN OPTIMIZATION STEP =====
            self.model.eval()
            self.merlin_optimizer.zero_grad()
            self.model.zero_grad()
            
            # 1. Get continuous masks from feature selectors
            continuous_mask_merlin = self.merlin(inputs)
            
            # 2. Apply L1 regularization (if configured) and normalize
            if self.l1_penalty_coefficient is not None:
                l1_penalty = self.l1_penalty_coefficient * torch.mean(torch.abs(continuous_mask_merlin))
                # continuous_mask_merlin = self.merlin.normalize_l1(continuous_mask_merlin, self.mask_size)
            else:
                l1_penalty = 0.0
            
            # 3. Apply continuous mask directly for training (preserves gradients)
            masked_inputs_merlin = self.merlin.apply_mask(inputs, continuous_mask_merlin)
            logits_merlin = self.model(masked_inputs_merlin)
            
            # 4. Calculate and backpropagate Merlin loss
            merlin_loss = self.criterion(logits_merlin, targets) + l1_penalty
            
            # Add reconstruction loss if enabled
            merlin_recon_loss = 0.0
            if use_reconstruction and saved_merlin_masks is not None:
                # OPTION 1: Compute BCE directly on continuous mask (before thresholding)
                if self.use_continuous_reconstruction:
                    # Apply positive example weighting if enabled
                    if self.weight_positive_examples:
                        # Calculate weights based on positive vs negative examples
                        num_pos = saved_merlin_masks.float().sum()
                        num_neg = saved_merlin_masks.numel() - num_pos
                        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
                        
                        # Use weighted BCE loss with logits
                        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
                        merlin_recon_loss = bce_loss_fn(continuous_mask_merlin, saved_merlin_masks.float())
                    else:
                        # Regular BCE loss with logits
                        bce_loss_fn = nn.BCEWithLogitsLoss()
                        merlin_recon_loss = bce_loss_fn(continuous_mask_merlin, saved_merlin_masks.float())
                # OPTION 2: Use straight-through estimator (STE) # NOTE: Not sure if this is learning correctly at the moment.
                else:
                    # Forward: discrete binary mask
                    binary_mask_merlin = self.merlin.get_binary_mask(continuous_mask_merlin)
                    # Backward: pretend it was continuous (STE trick)
                    binary_mask_st = binary_mask_merlin.detach() - continuous_mask_merlin.detach() + continuous_mask_merlin
                    
                    # Add positive example weighting if enabled
                    if self.weight_positive_examples:
                        # Calculate weights based on positive vs negative examples
                        num_pos = saved_merlin_masks.float().sum()
                        num_neg = saved_merlin_masks.numel() - num_pos
                        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
                        
                        # Use weighted BCE loss with binary_mask_st (maintaining the STE approach)
                        # We need to use F.binary_cross_entropy with the weight parameter
                        # rather than BCEWithLogitsLoss since we already have the transformed values
                        per_element_weight = torch.ones_like(saved_merlin_masks.float())
                        per_element_weight[saved_merlin_masks > 0] = pos_weight
                        
                        # Clamp values to valid range for BCE
                        binary_mask_st_clamped = torch.clamp(binary_mask_st.float(), 0.0, 1.0)


                        merlin_recon_loss = F.binary_cross_entropy(
                            binary_mask_st_clamped.float(),
                            saved_merlin_masks.float(),
                            weight=per_element_weight,
                            reduction='mean'
                        )
                    else:
                        # Regular BCE loss with clamped values
                        binary_mask_st_clamped = torch.clamp(binary_mask_st.float(), 0.0, 1.0)

                        merlin_recon_loss = F.binary_cross_entropy(
                            binary_mask_st_clamped.float(),
                            saved_merlin_masks.float(),
                            reduction='mean'
                        )
                
                # Add to overall loss
                merlin_loss = merlin_loss + recon_weight * merlin_recon_loss
                total_merlin_recon_loss += merlin_recon_loss.item()
                
            merlin_loss.backward()
            self.merlin_optimizer.step()
            
            # ===== MORGANA OPTIMIZATION STEP =====
            self.morgana_optimizer.zero_grad()
            self.model.zero_grad()
            
            # 1. Get continuous masks from Morgana
            continuous_mask_morgana = self.morgana(inputs)
            
            # 2. Apply L1 regularization (if configured) and normalize
            if self.l1_penalty_coefficient is not None:
                morgana_l1_penalty = self.l1_penalty_coefficient * torch.mean(torch.abs(continuous_mask_morgana))
                continuous_mask_morgana = self.morgana.normalize_l1(continuous_mask_morgana, self.mask_size)
            else:
                morgana_l1_penalty = 0.0
            
            # 3. Apply continuous mask directly for training (preserves gradients)
            masked_inputs_morgana = self.morgana.apply_mask(inputs, continuous_mask_morgana)
            logits_morgana = self.model(masked_inputs_morgana)
            
            # 4. Calculate Morgana loss
            morgana_loss = -self.morgana.criterion(logits_morgana, targets) + morgana_l1_penalty
            
            # Add reconstruction loss if enabled
            morgana_recon_loss = 0.0
            if use_reconstruction and saved_morgana_masks is not None:
                # OPTION 1: Compute BCE directly on continuous mask (before thresholding)
                if self.use_continuous_reconstruction:
                    # Apply positive example weighting if enabled
                    if self.weight_positive_examples:
                        # Calculate weights based on positive vs negative examples
                        num_pos = saved_morgana_masks.float().sum()
                        num_neg = saved_morgana_masks.numel() - num_pos
                        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
                        
                        # Use weighted BCE loss with logits
                        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
                        morgana_recon_loss = bce_loss_fn(continuous_mask_morgana, saved_morgana_masks.float())
                    else:
                        # Regular BCE loss with logits
                        bce_loss_fn = nn.BCEWithLogitsLoss()
                        morgana_recon_loss = bce_loss_fn(continuous_mask_morgana, saved_morgana_masks.float())
                # OPTION 2: Use straight-through estimator (STE) | # NOTE: Not sure if this is learning correctly at the moment.
                else:
                    print("Make sure you read the corresponding code again, and verify everything is fine. Morgana or Merlin might be off!")
                    # Forward: discrete binary mask
                    binary_mask_morgana = self.morgana.get_binary_mask(continuous_mask_morgana)
                    # Backward: pretend it was continuous (STE trick)
                    binary_mask_st = binary_mask_morgana.detach() - continuous_mask_morgana.detach() + continuous_mask_morgana
                    
                    # Add positive example weighting if enabled
                    if self.weight_positive_examples:
                        # Calculate weights based on positive vs negative examples
                        num_pos = saved_morgana_masks.float().sum()
                        num_neg = saved_morgana_masks.numel() - num_pos
                        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
                        
                        # Use weighted BCE loss with binary_mask_st (maintaining the STE approach)
                        # We need to use F.binary_cross_entropy with the weight parameter
                        # rather than BCEWithLogitsLoss since we already have the transformed values
                        per_element_weight = torch.ones_like(saved_morgana_masks.float())
                        per_element_weight[saved_morgana_masks > 0] = pos_weight
                        
                        morgana_recon_loss = F.binary_cross_entropy(
                            binary_mask_st.float(),
                            saved_morgana_masks.float(),
                            weight=per_element_weight,
                            reduction='mean'
                        )
                    else:
                        # Regular BCE loss
                        morgana_recon_loss = F.binary_cross_entropy(
                            binary_mask_st.float(),
                            saved_morgana_masks.float(),
                            reduction='mean'
                        )                
                # Add to overall loss
                morgana_loss = morgana_loss + recon_weight * morgana_recon_loss
                total_morgana_recon_loss += morgana_recon_loss.item()
            
            morgana_loss.backward()
            self.morgana_optimizer.step()
            
            # ===== CLASSIFIER OPTIMIZATION STEP =====
            self.optimizer.zero_grad()
            
            # For classifier training, use detached continuous masks
            detached_mask_merlin = self.merlin(inputs).detach()
            detached_mask_morgana = self.morgana(inputs).detach()
            
            # Binarize Masks
            binary_mask_merlin = self.merlin.get_binary_mask(detached_mask_merlin)
            binary_mask_morgana = self.morgana.get_binary_mask(detached_mask_morgana)
            
            # Apply masks to inputs
            masked_inputs_merlin = self.merlin.apply_mask(inputs, binary_mask_merlin)
            masked_inputs_morgana = self.morgana.apply_mask(inputs, binary_mask_morgana)
            
            # Forward through classifier
            logits_merlin = self.model(masked_inputs_merlin)
            logits_morgana = self.model(masked_inputs_morgana)
            
            # Calculate loss and update classifier
            arthur_loss = self.criterion(logits_merlin, targets) + self.gamma * self.morgana.criterion(logits_morgana, targets)
            arthur_loss.backward()
            self.optimizer.step()

            # Calculate evaluation metrics using binary masks
            batch_completeness = get_accuracy(logits_merlin, targets, mode="merlin", idk_class=self.num_classes)
            batch_soundness = get_accuracy(logits_morgana, targets, mode="morgana", idk_class=self.num_classes)                

            # Collect predictions for epoch-level balanced accuracy
            if 'cocologic' in self.dataset_name.lower():
                y_true_merlin.extend(targets.cpu().numpy().tolist())
                _, predicted_merlin = logits_merlin.max(1)
                y_pred_merlin.extend(predicted_merlin.cpu().numpy().tolist())

                # ADD: Collect Morgana predictions
                y_true_morgana.extend(targets.cpu().numpy().tolist())
                _, predicted_morgana = logits_morgana.max(1)
                y_pred_morgana.extend(predicted_morgana.cpu().numpy().tolist())


            # Track mask statistics if enabled
            if self.feature_selector_config.track_mask_statistics:
                merlin_stats = calculate_mask_overlap_statistics(inputs, binary_mask_merlin)
                morgana_stats = calculate_mask_overlap_statistics(inputs, binary_mask_morgana)
                
                merlin_batch_overlaps.append(merlin_stats['avg_mask_overlap'])
                morgana_batch_overlaps.append(morgana_stats['avg_mask_overlap'])
                merlin_feature_inclusions.append(merlin_stats['avg_feature_inclusion'])
                morgana_feature_inclusions.append(morgana_stats['avg_feature_inclusion'])
                batch_nonzeros.extend(merlin_stats['nonzero_counts'])
        
            # Update batch metrics
            total_loss += arthur_loss.item()
            total_completeness += batch_completeness
            total_soundness += batch_soundness
            batch_count += 1
            
            # Update progress bar with appropriate info
            if self.use_saved_masks:
                pbar.set_postfix({
                    'loss': f'{total_loss/batch_count:.4f}',
                    'comp': f'{100.*batch_completeness:.2f}%',
                    'sound': f'{100.*batch_soundness:.2f}%',
                    'merlin_recon': f'{merlin_recon_loss:.4f}',
                    'morgana_recon': f'{morgana_recon_loss:.4f}'
                })
            else:
                pbar.set_postfix({
                    'loss': f'{total_loss/batch_count:.4f}',
                    'comp': f'{100.*batch_completeness:.2f}%',
                    'sound': f'{100.*batch_soundness:.2f}%'
                })
    
        # Calculate epoch-level statistics
        avg_completeness = total_completeness / batch_count if batch_count > 0 else 0
        avg_soundness = total_soundness / batch_count if batch_count > 0 else 0
            
        # Only calculate these if we're tracking statistics
        avg_merlin_overlap = np.mean(merlin_batch_overlaps) if merlin_batch_overlaps else 0
        avg_morgana_overlap = np.mean(morgana_batch_overlaps) if morgana_batch_overlaps else 0
        avg_merlin_inclusion = np.mean(merlin_feature_inclusions) if merlin_feature_inclusions else 0
        avg_morgana_inclusion = np.mean(morgana_feature_inclusions) if morgana_feature_inclusions else 0
        avg_nonzeros = np.mean(batch_nonzeros) if batch_nonzeros else 0
            
        # Compute epoch metrics
        metrics = {
            'train_loss': total_loss / batch_count,
            'train_completeness': 100. * avg_completeness,
            'train_soundness': 100. * avg_soundness,
            'merlin_feature_overlap': avg_merlin_overlap,
            'morgana_feature_overlap': avg_morgana_overlap,
            'merlin_feature_inclusion': avg_merlin_inclusion,
            'morgana_feature_inclusion': avg_morgana_inclusion,
            'nonzero_features': avg_nonzeros
        }
        
        if 'cocologic' in self.dataset_name.lower():
            metrics['train_balanced_completeness'] = 100. * balanced_accuracy_score(y_true_merlin, y_pred_merlin)
            metrics['train_balanced_soundness'] = 100. * get_balanced_soundness_cocologic(
                y_pred_morgana, y_true_morgana, self.num_classes)

        if use_reconstruction:
            metrics.update({
                'merlin_recon_loss': total_merlin_recon_loss / batch_count if batch_count > 0 else 0,
                'morgana_recon_loss': total_morgana_recon_loss / batch_count if batch_count > 0 else 0
            })
        
        return metrics

    def validate(self):
        """Run validation based on specified approach
        
        Returns:
            dict: Dictionary containing validation metrics
        """
        if self.approach == "regular":
            return self._validate_regular()
        elif self.approach == "sfw":
            return self._validate_sfw()
        elif self.approach == "nn":
            return self._validate_nn()
        elif self.approach == "posthoc":
            raise NotImplementedError("Post-hoc validation not implemented yet")
        else:
            raise ValueError(f"Unknown approach: {self.approach}")

    def _validate_regular(self):
        """Regular validation approach
        
        Returns:
            dict: Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        # Progress bar for validation
        pbar = tqdm(self.val_loader, desc='Validating')
        
        with torch.no_grad():
            for inputs, targets in pbar:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                y_true.extend(targets.cpu().numpy().tolist())
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                y_pred.extend(predicted.cpu().numpy().tolist())
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss/len(self.val_loader):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # Compute validation metrics
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_acc': 100. * correct / total,
            'val_balanced_acc': 100. * balanced_accuracy_score(y_true, y_pred)
        }
        
        return metrics

    def _validate_sfw(self):
        """Validate model using SFW Merlin-Arthur approach
        
        Returns:
            dict: Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_completeness = 0
        total_soundness = 0
        batch_count = 0
        
        # Track overlap statistics (only initialize if needed)
        merlin_batch_overlaps = []
        morgana_batch_overlaps = []
        batch_nonzeros = []
        
        # Progress bar for validation
        pbar = tqdm(self.val_loader, desc='Validating with Merlin-Arthur')
        
        for inputs, targets in pbar:
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Temporarily enable gradients for mask optimization
            with torch.enable_grad():
                # Optimize masks (in eval mode) - requires gradients
                continuous_mask_merlin = self.merlin(inputs, targets, self.model)
                continuous_mask_morgana = self.morgana(inputs, targets, self.model)
                
                # Convert to binary masks using top-k selection
                binary_mask_merlin = self.merlin.get_binary_mask(continuous_mask_merlin)
                binary_mask_morgana = self.morgana.get_binary_mask(continuous_mask_morgana)
            
            # Calculate overlap statistics ONLY if tracking is enabled
            if self.feature_selector_config.track_mask_statistics:
                merlin_stats = calculate_mask_overlap_statistics(inputs, binary_mask_merlin)
                morgana_stats = calculate_mask_overlap_statistics(inputs, binary_mask_morgana)
                
                # Store statistics for epoch averaging
                merlin_batch_overlaps.append(merlin_stats['avg_mask_overlap'])
                morgana_batch_overlaps.append(morgana_stats['avg_mask_overlap'])
                batch_nonzeros.extend(merlin_stats['nonzero_counts'])
            
            # Disable gradients for the rest of the validation process
            with torch.no_grad():
                # Apply masks and get predictions
                masked_inputs_merlin = self.merlin.apply_mask(inputs, binary_mask_merlin)
                masked_inputs_morgana = self.morgana.apply_mask(inputs, binary_mask_morgana)
                
                logits_merlin = self.model(masked_inputs_merlin)
                logits_morgana = self.model(masked_inputs_morgana)
                
                # Calculate loss
                merlin_loss = self.criterion(logits_merlin, targets)
                morgana_loss = self.morgana.criterion(logits_morgana, targets)
                loss = merlin_loss + self.gamma * morgana_loss
                
                # Update metrics
                total_loss += loss.item()
                
                # Calculate accuracies (completeness and soundness)
                batch_completeness = get_accuracy(logits_merlin, targets, mode="merlin", idk_class=self.num_classes)
                batch_soundness = get_accuracy(logits_morgana, targets, mode="morgana", idk_class=self.num_classes)
                
                # Accumulate for epoch average
                total_completeness += batch_completeness
                total_soundness += batch_soundness
                batch_count += 1
                
                # Update progress bar with appropriate info
                if self.feature_selector_config.track_mask_statistics:
                    pbar.set_postfix({
                        'loss': f'{total_loss/batch_count:.4f}',
                        'comp': f'{100.*batch_completeness:.2f}%', 
                        'sound': f'{100.*batch_soundness:.2f}%',
                        'M-ovlp': f'{merlin_stats["avg_mask_overlap"]:.1f}%',
                        'M-incl': f'{merlin_stats["avg_feature_inclusion"]:.1f}%'
                    })
                else:
                    pbar.set_postfix({
                        'loss': f'{total_loss/batch_count:.4f}',
                        'comp': f'{100.*batch_completeness:.2f}%', 
                        'sound': f'{100.*batch_soundness:.2f}%'
                    })
        
        # Calculate epoch-level metrics
        avg_completeness = total_completeness / batch_count if batch_count > 0 else 0
        avg_soundness = total_soundness / batch_count if batch_count > 0 else 0
        
        # Only calculate these if we're tracking statistics
        avg_merlin_overlap = np.mean(merlin_batch_overlaps) if merlin_batch_overlaps else 0
        avg_morgana_overlap = np.mean(morgana_batch_overlaps) if morgana_batch_overlaps else 0
        avg_nonzeros = np.mean(batch_nonzeros) if batch_nonzeros else 0
                
        # Compute validation metrics
        metrics = {
            'val_loss': total_loss / batch_count,
            'val_acc': 100. * avg_completeness,  # Keep val_acc for backward compatibility
            'val_completeness': 100. * avg_completeness,
            'val_soundness': 100. * avg_soundness,
        }
        
        # Only add these metrics if tracking statistics
        if self.feature_selector_config.track_mask_statistics:
            metrics.update({
                'merlin_feature_overlap': avg_merlin_overlap,
                'morgana_feature_overlap': avg_morgana_overlap,
                'nonzero_features': avg_nonzeros
            })
        
        return metrics

    def _validate_nn(self):
        """Validate model using neural network feature selectors
        
        Returns:
            dict: Dictionary containing validation metrics
        """
        self.model.eval()
        self.merlin.eval()
        self.morgana.eval()
        
        total_loss = 0
        total_completeness = 0
        total_soundness = 0
        total_merlin_recon_loss = 0
        total_morgana_recon_loss = 0
        batch_count = 0
        
        # Track overlap statistics (only initialize if needed)
        merlin_batch_overlaps = []
        morgana_batch_overlaps = []
        merlin_feature_inclusions = []
        morgana_feature_inclusions = []
        batch_nonzeros = []

        # Add balanced completeness tracking ONLY for CoCoLogic
        if 'cocologic' in self.dataset_name.lower():
            y_true_merlin = []
            y_pred_merlin = []
            y_true_morgana = []  # ADD this
            y_pred_morgana = []  # ADD this


        # Use reconstruction regularization if enabled (use the class variable)
        use_reconstruction = self.use_saved_masks and hasattr(self, 'saved_val_merlin_masks')
        
        # Progress bar for validation
        pbar = tqdm(self.val_loader, desc='Validating with Neural Feature Selectors')
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                # Handle both regular data tuples and indexed data
                if use_reconstruction and len(batch_data) == 3:  # (inputs, targets, indices)
                    inputs, targets, indices = batch_data
                    # indices = indices.to(self.device)
                else:
                    inputs, targets = batch_data
                    indices = None
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Load saved masks for this batch if reconstruction is enabled
                if use_reconstruction and indices is not None and self.saved_val_merlin_masks is not None:
                    # Get masks corresponding to the specific validation sample indices
                    saved_merlin_masks = self.saved_val_merlin_masks[indices].to(self.device)
                    saved_morgana_masks = self.saved_val_morgana_masks[indices].to(self.device)
                else:
                    saved_merlin_masks = None
                    saved_morgana_masks = None
                
                # Generate masks
                continuous_mask_merlin = self.merlin(inputs)
                continuous_mask_morgana = self.morgana(inputs)
                
                # Calculate reconstruction loss if enabled
                merlin_recon_loss = 0.0
                morgana_recon_loss = 0.0
                
                if use_reconstruction and saved_merlin_masks is not None:
                    # OPTION 1: Compute BCE directly on continuous mask (before thresholding)
                    if self.feature_selector_config.use_continuous_reconstruction:
                        # Apply positive example weighting if enabled
                        if self.feature_selector_config.weight_positive_examples:
                            # Calculate weights for Merlin
                            num_pos_merlin = saved_merlin_masks.float().sum()
                            num_neg_merlin = saved_merlin_masks.numel() - num_pos_merlin
                            pos_weight_merlin = num_neg_merlin / num_pos_merlin if num_pos_merlin > 0 else 1.0
                            
                            # Calculate weights for Morgana
                            num_pos_morgana = saved_morgana_masks.float().sum()
                            num_neg_morgana = saved_morgana_masks.numel() - num_pos_morgana
                            pos_weight_morgana = num_neg_morgana / num_pos_morgana if num_pos_morgana > 0 else 1.0
                            
                            # Use weighted BCE loss for both
                            bce_loss_fn_merlin = nn.BCEWithLogitsLoss(pos_weight=pos_weight_merlin.to(self.device))
                            bce_loss_fn_morgana = nn.BCEWithLogitsLoss(pos_weight=pos_weight_morgana.to(self.device))
                            
                            merlin_recon_loss = bce_loss_fn_merlin(continuous_mask_merlin, saved_merlin_masks.float())
                            morgana_recon_loss = bce_loss_fn_morgana(continuous_mask_morgana, saved_morgana_masks.float())
                        else:
                            # Use BCEWithLogitsLoss instead of sigmoid + BCE
                            bce_loss_fn = nn.BCEWithLogitsLoss()
                            merlin_recon_loss = bce_loss_fn(continuous_mask_merlin, saved_merlin_masks.float())
                            morgana_recon_loss = bce_loss_fn(continuous_mask_morgana, saved_morgana_masks.float())
                    # OPTION 2: Use straight-through estimator approach 
                    else:
                        binary_mask_merlin = self.merlin.get_binary_mask(continuous_mask_merlin)
                        binary_mask_morgana = self.morgana.get_binary_mask(continuous_mask_morgana)
                        
                        # Need to create STE versions like in training
                        binary_mask_st_merlin = binary_mask_merlin.detach() - continuous_mask_merlin.detach() + continuous_mask_merlin
                        binary_mask_st_morgana = binary_mask_morgana.detach() - continuous_mask_morgana.detach() + continuous_mask_morgana
                        
                        # Clamp values to valid range for BCE (same as training)
                        binary_mask_st_merlin_clamped = torch.clamp(binary_mask_st_merlin.float(), 0.0, 1.0)
                        binary_mask_st_morgana_clamped = torch.clamp(binary_mask_st_morgana.float(), 0.0, 1.0)
                        
                        # Use the clamped STE versions for loss calculation
                        if self.weight_positive_examples:
                            # Calculate weights for Merlin
                            num_pos_merlin = saved_merlin_masks.float().sum()
                            num_neg_merlin = saved_merlin_masks.numel() - num_pos_merlin
                            pos_weight_merlin = num_neg_merlin / num_pos_merlin if num_pos_merlin > 0 else 1.0
                            
                            # Calculate weights for Morgana
                            num_pos_morgana = saved_morgana_masks.float().sum()
                            num_neg_morgana = saved_morgana_masks.numel() - num_pos_morgana
                            pos_weight_morgana = num_neg_morgana / num_pos_morgana if num_pos_morgana > 0 else 1.0
                            
                            # Create per-element weights
                            merlin_weights = torch.ones_like(saved_merlin_masks.float())
                            merlin_weights[saved_merlin_masks > 0] = pos_weight_merlin
                            
                            morgana_weights = torch.ones_like(saved_morgana_masks.float())
                            morgana_weights[saved_morgana_masks > 0] = pos_weight_morgana
                            
                            # Use weighted BCE with the STE masks
                            merlin_recon_loss = F.binary_cross_entropy(
                                binary_mask_st_merlin_clamped,
                                saved_merlin_masks.float(),
                                weight=merlin_weights,
                                reduction='mean'
                            )
                            
                            morgana_recon_loss = F.binary_cross_entropy(
                                binary_mask_st_morgana_clamped,
                                saved_morgana_masks.float(),
                                weight=morgana_weights,
                                reduction='mean'
                            )
                        else:
                            # Regular BCE with the STE masks (no weighting)
                            merlin_recon_loss = F.binary_cross_entropy(
                                binary_mask_st_merlin_clamped,
                                saved_merlin_masks.float(),
                                reduction='mean'
                            )
                            
                            morgana_recon_loss = F.binary_cross_entropy(
                                binary_mask_st_morgana_clamped,
                                saved_morgana_masks.float(),
                                reduction='mean'
                            )
                
                # Accumulate reconstruction losses
                total_merlin_recon_loss += merlin_recon_loss.item() if hasattr(merlin_recon_loss, 'item') else merlin_recon_loss
                total_morgana_recon_loss += morgana_recon_loss.item() if hasattr(morgana_recon_loss, 'item') else morgana_recon_loss
                
                # Convert to binary masks
                binary_mask_merlin = self.merlin.get_binary_mask(continuous_mask_merlin)
                binary_mask_morgana = self.morgana.get_binary_mask(continuous_mask_morgana)
                
                # Calculate overlap statistics ONLY if tracking is enabled
                if self.feature_selector_config.track_mask_statistics:
                    merlin_stats = calculate_mask_overlap_statistics(inputs, binary_mask_merlin)
                    morgana_stats = calculate_mask_overlap_statistics(inputs, binary_mask_morgana)
                    
                    # Store statistics for epoch averaging
                    merlin_batch_overlaps.append(merlin_stats['avg_mask_overlap'])
                    morgana_batch_overlaps.append(morgana_stats['avg_mask_overlap'])
                    merlin_feature_inclusions.append(merlin_stats['avg_feature_inclusion'])
                    morgana_feature_inclusions.append(morgana_stats['avg_feature_inclusion'])
                    batch_nonzeros.extend(merlin_stats['nonzero_counts'])
                
                # Apply masks and get predictions
                masked_inputs_merlin = self.merlin.apply_mask(inputs, binary_mask_merlin)
                masked_inputs_morgana = self.morgana.apply_mask(inputs, binary_mask_morgana)
                
                logits_merlin = self.model(masked_inputs_merlin)
                logits_morgana = self.model(masked_inputs_morgana)
                
                # Calculate loss
                merlin_loss = self.criterion(logits_merlin, targets)
                morgana_loss = self.criterion(logits_morgana, targets)
                loss = merlin_loss + self.gamma * morgana_loss
                
                # Update metrics
                total_loss += loss.item()
                
                # Calculate accuracies (completeness and soundness)
                batch_completeness = get_accuracy(logits_merlin, targets, mode="merlin", idk_class=self.num_classes)
                batch_soundness = get_accuracy(logits_morgana, targets, mode="morgana", idk_class=self.num_classes)
                

                # Collect predictions for balanced accuracy ONLY for CoCoLogic
                if 'cocologic' in self.dataset_name.lower():
                    y_true_merlin.extend(targets.cpu().numpy().tolist())
                    _, predicted_merlin = logits_merlin.max(1)
                    y_pred_merlin.extend(predicted_merlin.cpu().numpy().tolist())

                    y_true_morgana.extend(targets.cpu().numpy().tolist())
                    _, predicted_morgana = logits_morgana.max(1)
                    y_pred_morgana.extend(predicted_morgana.cpu().numpy().tolist())


                # Accumulate for epoch average
                total_completeness += batch_completeness
                total_soundness += batch_soundness
                batch_count += 1
                
                # Update progress bar with appropriate info
                if self.feature_selector_config.track_mask_statistics:
                    pbar.set_postfix({
                        'loss': f'{total_loss/batch_count:.4f}',
                        'comp': f'{100.*batch_completeness:.2f}%',
                        'sound': f'{100.*batch_soundness:.2f}%',
                        'M-ovlp': f'{merlin_stats["avg_mask_overlap"]:.1f}%',
                        'M-incl': f'{merlin_stats["avg_feature_inclusion"]:.1f}%'
                    })
                elif self.use_saved_masks:
                    pbar.set_postfix({
                        'loss': f'{total_loss/batch_count:.4f}',
                        'comp': f'{100.*batch_completeness:.2f}%',
                        'sound': f'{100.*batch_soundness:.2f}%',
                        'merlin_recon': f'{merlin_recon_loss:.4f}',
                        'morgana_recon': f'{morgana_recon_loss:.4f}'
                    })
                else:
                    pbar.set_postfix({
                        'loss': f'{total_loss/batch_count:.4f}',
                        'comp': f'{100.*batch_completeness:.2f}%',
                        'sound': f'{100.*batch_soundness:.2f}%'
                    })
        
        # Calculate epoch-level metrics
        avg_completeness = total_completeness / batch_count if batch_count > 0 else 0
        avg_soundness = total_soundness / batch_count if batch_count > 0 else 0
        
        # Only calculate these if we're tracking statistics
        avg_merlin_overlap = np.mean(merlin_batch_overlaps) if merlin_batch_overlaps else 0
        avg_morgana_overlap = np.mean(morgana_batch_overlaps) if morgana_batch_overlaps else 0
        avg_merlin_inclusion = np.mean(merlin_feature_inclusions) if merlin_feature_inclusions else 0
        avg_morgana_inclusion = np.mean(morgana_feature_inclusions) if morgana_feature_inclusions else 0
        avg_nonzeros = np.mean(batch_nonzeros) if batch_nonzeros else 0
        
        # Compute validation metrics
        metrics = {
            'val_loss': total_loss / batch_count,
            'val_acc': 100. * avg_completeness,  # Keep val_acc for backward compatibility
            'val_completeness': 100. * avg_completeness,
            'val_soundness': 100. * avg_soundness,
        }
        
        if 'cocologic' in self.dataset_name.lower():
            metrics['val_balanced_completeness'] = 100. * balanced_accuracy_score(y_true_merlin, y_pred_merlin)
            metrics['val_balanced_soundness'] = 100. * get_balanced_soundness_cocologic(y_pred_morgana, y_true_morgana, self.num_classes)

        # Add reconstruction loss metrics if enabled
        if use_reconstruction:
            metrics.update({
                'val_merlin_recon_loss': total_merlin_recon_loss / batch_count if batch_count > 0 else 0,
                'val_morgana_recon_loss': total_morgana_recon_loss / batch_count if batch_count > 0 else 0
            })
        
        # Only add these metrics if tracking statistics
        if self.feature_selector_config.track_mask_statistics:
            metrics.update({
                'merlin_feature_overlap': avg_merlin_overlap,
                'morgana_feature_overlap': avg_morgana_overlap,
                'merlin_feature_inclusion': avg_merlin_inclusion,
                'morgana_feature_inclusion': avg_morgana_inclusion,
                'nonzero_features': avg_nonzeros
            })
        
        return metrics
    
    def _train_unet(self):
        """Train model using U-Net approach
        
        Returns:
            dict: Dictionary containing training metrics
        """
        print(f"\nStarting Merlin-Arthur training with U-Net feature selectors for {self.epochs} epochs...")
            
        best_metrics = {
            'best_combined_metric': 0,
            'best_epoch': -1,
            'val_loss': float('inf'),
            'val_completeness': 0,
            'val_soundness': 0,
            'train_completeness': 0,
            'train_soundness': 0,
            'train_loss': float('inf'),
            'train_balanced_completeness': 0,
            'val_balanced_completeness': 0        
        }

        # Simple tracking for true best completeness/soundness
        best_completeness = 0
        best_soundness = 0
        best_comp_epoch = -1
        best_sound_epoch = -1

        # ADD: For CoCoLogic datasets
        if 'cocologic' in self.dataset_name.lower():
            best_balanced_completeness = 0
            best_balanced_comp_epoch = -1
            best_balanced_soundness = 0
            best_balanced_sound_epoch = -1


        no_improvement = 0  # ADD: Counter for early stopping
        best_model_state = None  # Store the best model state

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train and validate
            train_metrics = self._train_epoch_unet()
            val_metrics = self._validate_unet(self.val_loader)
            
            # Update validation metrics to include completeness and soundness
            if 'val_completeness' not in val_metrics:
                # For backward compatibility, use val_acc as completeness if not already present
                val_metrics['val_completeness'] = val_metrics['acc']
                val_metrics['val_soundness'] = 0  # Default if not present
            
            # Log metrics to wandb if enabled
            if self.logger is not None:
                metrics_dict = {
                    'epoch': epoch,
                    'train/loss': train_metrics['train_loss'],
                    'train/completeness': train_metrics['train_completeness'],
                    'train/soundness': train_metrics['train_soundness'],
                    'val/loss': val_metrics['loss'],
                    'val/completeness': val_metrics['completeness'],
                    'val/soundness': val_metrics['soundness'],
                    'mask/sparsity': self.mask_size,
                }

            # ADD: Balanced metrics for CoCoLogic
            if 'cocologic' in self.dataset_name.lower():
                if 'train_balanced_completeness' in train_metrics:
                    metrics_dict['train/balanced_completeness'] = train_metrics['train_balanced_completeness']
                if 'val_balanced_completeness' in val_metrics:
                    metrics_dict['val/balanced_completeness'] = val_metrics['val_balanced_completeness']
                if 'train_balanced_soundness' in train_metrics:
                    metrics_dict['train/balanced_soundness'] = train_metrics['train_balanced_soundness']
                if 'val_balanced_soundness' in val_metrics:
                    metrics_dict['val/balanced_soundness'] = val_metrics['val_balanced_soundness']

            if self.logger is not None:
                self.logger.log(metrics_dict, step=epoch)


            # Consolidated Epoch Summary
            print("\n" + "=" * 60)
            print("EPOCH SUMMARY")
            print("=" * 60)
            print(f"Epoch {epoch+1}/{self.epochs}")
            print("-" * 60)
            print("Training Statistics:")
            print(f"   Loss            : {train_metrics['train_loss']:.4f}")
            print(f"   Completeness    : {train_metrics['train_completeness']:.2f}%")
            if 'cocologic' in self.dataset_name.lower() and 'train_balanced_completeness' in train_metrics:
                print(f"   Balanced Comp.  : {train_metrics['train_balanced_completeness']:.2f}%")
            print(f"   Soundness       : {train_metrics['train_soundness']:.2f}%")
            if 'cocologic' in self.dataset_name.lower() and 'train_balanced_soundness' in train_metrics:
                print(f"   Balanced Sound. : {train_metrics['train_balanced_soundness']:.2f}%")
            print("-" * 60)
            print("Validation Statistics:")
            print(f"   Loss            : {val_metrics['loss']:.4f}")
            print(f"   Completeness    : {val_metrics['completeness']:.2f}%")
            if 'cocologic' in self.dataset_name.lower() and 'val_balanced_completeness' in val_metrics:
                print(f"   Balanced Comp.  : {val_metrics['val_balanced_completeness']:.2f}%")
            if 'cocologic' in self.dataset_name.lower() and 'val_balanced_soundness' in val_metrics:
                print(f"   Balanced Sound. : {val_metrics['val_balanced_soundness']:.2f}%")
            print(f"   Soundness       : {val_metrics['soundness']:.2f}%")
            print("-" * 60)

            # Update best metrics and save model
            if val_metrics['soundness'] > 90:
                # If soundness threshold is met, add a large bonus to ensure it's better than sub-90 models
                # but still maintains the comp+sound ordering among models above 90%
                combined_metric = val_metrics['completeness'] + val_metrics['soundness'] + 100
            else:
                combined_metric = val_metrics['completeness'] + val_metrics['soundness']
                
            if combined_metric > best_metrics['best_combined_metric']:
                best_metrics.update({
                    'best_combined_metric': combined_metric,
                    'best_epoch': epoch,
                    'val_loss': val_metrics['loss'],
                    'val_completeness': val_metrics['completeness'],
                    'val_soundness': val_metrics['soundness'],
                    'train_completeness': train_metrics['train_completeness'],
                    'train_soundness': train_metrics['train_soundness'],
                    'train_loss': train_metrics['train_loss']
                })

                # ADD: CoCoLogic balanced metrics to best_metrics
                if 'cocologic' in self.dataset_name.lower():
                    if 'train_balanced_completeness' in train_metrics:
                        best_metrics['train_balanced_completeness'] = train_metrics['train_balanced_completeness']
                    if 'val_balanced_completeness' in val_metrics:
                        best_metrics['val_balanced_completeness'] = val_metrics['val_balanced_completeness']

                no_improvement = 0  # ADD: Reset counter on improvement
 
                
                if self.save_model:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'merlin_state_dict': self.merlin.state_dict(),
                        'morgana_state_dict': self.morgana.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'merlin_optimizer_state_dict': self.merlin_optimizer.state_dict(),
                        'morgana_optimizer_state_dict': self.morgana_optimizer.state_dict(),
                        **best_metrics
                    }

                    # Determine checkpoint directory based on pretrained_path if available
                    if self.pretrained_path:
                        # Use the provided pretrained_path
                        checkpoint_dir = os.path.join(
                            self.pretrained_path,
                            f'{self.dataset_name}',
                            f'unet_approach',
                            f"mask_size_{self.mask_size}",
                            f"{self.model_name}_nn_mask_{self.mask_size}"  # Include model name before nn_mask
                        )
                    else:
                        raise ValueError("pretrained_path must be specified to save the best model checkpoint.")

                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                        
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Updated best model checkpoint with combined metric: {combined_metric:.2f}")
            else:
                no_improvement += 1  # ADD: Increment counter if no improvement
                print(f"No improvement in combined metric for {no_improvement} epoch(s).")



            # ADD: Simple best tracking AFTER the existing best metrics logic
            if val_metrics['completeness'] > best_completeness:
                best_completeness = val_metrics['completeness']
                best_comp_epoch = epoch
                
            if val_metrics['soundness'] > best_soundness:
                best_soundness = val_metrics['soundness']
                best_sound_epoch = epoch

            if 'cocologic' in self.dataset_name.lower() and 'val_balanced_completeness' in val_metrics:
                if val_metrics['val_balanced_completeness'] > best_balanced_completeness:
                    best_balanced_completeness = val_metrics['val_balanced_completeness']
                    best_balanced_comp_epoch = epoch

            if 'cocologic' in self.dataset_name.lower() and 'val_balanced_soundness' in val_metrics:
                if val_metrics['val_balanced_soundness'] > best_balanced_soundness:
                    best_balanced_soundness = val_metrics['val_balanced_soundness']
                    best_balanced_sound_epoch = epoch

            # ADD: Early stopping check
            if self.early_stopping and no_improvement >= self.patience:
                print(f"Early stopping triggered after {no_improvement} epochs with no improvement.")
                break

         # Print final summary with best metrics
        print(f"\nTraining completed!")
        # Print final metrics with best model
        print("\nFinal evaluation with best model:\n")
        print(f"Best combined metric at epoch {best_metrics['best_epoch']+1}")
        print(f"Best validation completeness: {best_metrics['val_completeness']:.2f}%")
        print(f"Best validation soundness: {best_metrics['val_soundness']:.2f}%")
        if 'cocologic' in self.dataset_name.lower():
            print(f"Best balanced completeness (overall): {best_balanced_completeness:.2f}% at epoch {best_balanced_comp_epoch+1}")
            print(f"Best balanced soundness (overall): {best_balanced_soundness:.2f}% at epoch {best_balanced_sound_epoch+1}")

        

        # test_metrics = self._validate_unet(self.val_loader)
        # print(f"Best test completeness: {test_metrics['completeness']:.2f}%")
        # print(f"Best test soundness: {test_metrics['soundness']:.2f}%")

        # Enhanced wandb logging
        if self.logger is not None:
            final_metrics = {
                'val/best_combined_metric': best_metrics['best_combined_metric'],
                'val/best_loss': best_metrics['val_loss'],
                'val/best_epoch': best_metrics['best_epoch'],
                'val/best_completeness': best_metrics['val_completeness'],
                'val/best_soundness': best_metrics['val_soundness'],
                # 'test/best_completeness': test_metrics['completeness'],
                # 'test/best_soundness': test_metrics['soundness'],
                
                # ADD: True best metrics
                'best_overall/val_completeness': best_completeness,
                'best_overall/val_soundness': best_soundness,
                'best_overall/comp_epoch': best_comp_epoch+1,
                'best_overall/sound_epoch': best_sound_epoch+1
            }
            
            # ADD: CoCoLogic balanced metrics
            if 'cocologic' in self.dataset_name.lower():
                if 'train_balanced_completeness' in best_metrics:
                    final_metrics['best/train_balanced_completeness'] = best_metrics['train_balanced_completeness']
                if 'val_balanced_completeness' in best_metrics:
                    final_metrics['best/val_balanced_completeness'] = best_metrics['val_balanced_completeness']
                final_metrics['best_overall/val_balanced_completeness'] = best_balanced_completeness
                final_metrics['best_overall/balanced_comp_epoch'] = best_balanced_comp_epoch+1
                final_metrics['best_overall/val_balanced_soundness'] = best_balanced_soundness
                final_metrics['best_overall/balanced_sound_epoch'] = best_balanced_sound_epoch+1
            
            self.logger.log(final_metrics)
        
        return best_metrics

    def _train_epoch_unet(self):
        """Train model for one epoch using U-Net approach
        
        Returns:
            dict: Dictionary containing training metrics
        """
        self.model.train()
        self.merlin.train()
        self.morgana.train()

        total_loss = 0
        total_completeness = 0
        total_soundness = 0
        batch_count = 0

        # ADD: For CoCoLogic datasets
        if 'cocologic' in self.dataset_name.lower():
            y_true_merlin = []
            y_pred_merlin = []
            y_true_morgana = []
            y_pred_morgana = []


         # Progress bar for training
        pbar = tqdm(self.train_loader, desc='Training with Merlin-Arthur (U-Net)')
        
        for inputs, targets in pbar:
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()
            
            # Step 1: Optimize masks using learnable feature selectors
            continuous_mask_merlin = self._optimize_unet(inputs, targets, self.merlin, self.merlin_optimizer, steps=1)
            continuous_mask_morgana = self._optimize_unet(inputs, targets, self.morgana, self.morgana_optimizer, steps=1)
            
            # Step 2: Convert to binary masks using top-k selection
            binary_mask_merlin = self.merlin.get_binary_mask(continuous_mask_merlin)
            binary_mask_morgana = self.morgana.get_binary_mask(continuous_mask_morgana)

            # Step 3: Apply mask and compute logits

            masked_inputs_merlin = self.merlin.apply_mask(inputs, binary_mask_merlin)
            masked_inputs_morgana = self.morgana.apply_mask(inputs, binary_mask_morgana)

            self.model.eval()  # NOTE: Need to be in eval mode to prevent batchnorm from updating

            logits_merlin = self.model(masked_inputs_merlin)
            logits_morgana = self.model(masked_inputs_morgana)

            # Step 4: Calculate losses
            # Merlin and Morgana losses
            merlin_loss = self.merlin.criterion(logits_merlin, targets)
            morgana_loss = self.morgana.criterion(logits_morgana, targets)
            
            # Combined loss for the classifier
            loss = merlin_loss + self.gamma * morgana_loss

            loss.backward()

            # Update optimizer
            self.optimizer.step()

            self.optimizer.zero_grad()
            self.merlin_optimizer.zero_grad()
            self.morgana_optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()

            # Calculate accuracies (completeness and soundness)
            batch_completeness = get_accuracy(logits_merlin, targets, mode="merlin", idk_class=self.num_classes)
            batch_soundness = get_accuracy(logits_morgana, targets, mode="morgana", idk_class=self.num_classes)

            # ADD: Collect predictions for balanced accuracy
            if 'cocologic' in self.dataset_name.lower():
                y_true_merlin.extend(targets.cpu().numpy().tolist())
                _, predicted_merlin = logits_merlin.max(1)
                y_pred_merlin.extend(predicted_merlin.cpu().numpy().tolist())

                y_true_morgana.extend(targets.cpu().numpy().tolist())
                _, predicted_morgana = logits_morgana.max(1)
                y_pred_morgana.extend(predicted_morgana.cpu().numpy().tolist())

            
            # Accumulate for epoch average
            total_completeness += batch_completeness
            total_soundness += batch_soundness
            batch_count += 1   

            # Update progress bar with overlap info and accuracy
            pbar.set_postfix({
                'loss': f'{total_loss/batch_count:.4f}',
                'comp': f'{100.*batch_completeness:.2f}%',
                'sound': f'{100.*batch_soundness:.2f}%'
            })    

        avg_completeness = total_completeness / batch_count if batch_count > 0 else 0
        avg_soundness = total_soundness / batch_count if batch_count > 0 else 0

        # Compute epoch metrics
        metrics = {
            'train_loss': total_loss / batch_count,
            'train_completeness': 100. * avg_completeness,
            'train_soundness': 100. * avg_soundness
        }

        # ADD: Calculate balanced accuracy for CoCoLogic
        if 'cocologic' in self.dataset_name.lower():
            metrics['train_balanced_completeness'] = 100. * balanced_accuracy_score(y_true_merlin, y_pred_merlin)
            metrics['train_balanced_soundness'] = 100. * get_balanced_soundness_cocologic(
                y_pred_morgana, y_true_morgana, self.num_classes)

        return metrics

    def _optimize_unet(self, inputs, targets, feature_selector, optimizer, steps=1):
        """
        Single optimization step for the U-Net (Merlin/Morgana)
        """
        self.model.eval()

        for _ in range(steps):
            continuous_mask = feature_selector(inputs)
            continuous_mask = feature_selector.normalize_l1(continuous_mask, self.mask_size)

            l1_penalty = self.l1_penalty_coefficient * torch.mean(torch.abs(continuous_mask))
            # l2_penalty = self.l2_penalty_coefficient * torch.mean(torch.square(continuous_mask))

            tv_norm = torch.sum(torch.abs(continuous_mask[:, :, :, :-1] - continuous_mask[:, :, :, 1:]) ** 2) + torch.sum(
                        torch.abs(continuous_mask[:, :, :-1, :] - continuous_mask[:, :, 1:, :]) ** 2) 
            tv_norm = tv_norm / (continuous_mask.shape[0])
            # tv_penalty = self.tv_penalty_coefficient * tv_norm
 
            
            masked_inputs = feature_selector.apply_mask(inputs, continuous_mask)
            logits = self.model(masked_inputs)

            if feature_selector.mode == "merlin":
                loss = feature_selector.criterion(logits, targets) + l1_penalty # + l2_penalty # + tv_penalty
            elif feature_selector.mode == "morgana":
                loss = -feature_selector.criterion(logits, targets) + l1_penalty # + l2_penalty # + tv_penalty

        loss.backward()
        optimizer.step()
        self.optimizer.zero_grad() # Arthur optimizer
        optimizer.zero_grad() # Feature selector optimizer

        return continuous_mask

    def _validate_unet(self, loader: DataLoader):
        """Validate model using U-Net approach
        
        Returns:
            dict: Dictionary containing validation metrics
        """
        self.model.eval()
        self.merlin.eval()
        self.morgana.eval()

        total_loss = 0
        total_completeness = 0
        total_soundness = 0
        batch_count = 0

        # ADD: For CoCoLogic datasets
        if 'cocologic' in self.dataset_name.lower():
            y_true_merlin = []
            y_pred_merlin = []
            y_true_morgana = []
            y_pred_morgana = []

        
        # Progress bar for validation
        pbar = tqdm(loader, desc='Validating with Merlin-Arthur' if loader == self.val_loader else 'Testing with Merlin-Arthur')
        
        for inputs, targets in pbar:
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()

            # Disable gradients for the validation process
            with torch.no_grad():
                continuous_mask_merlin = self.merlin(inputs)
                continuous_mask_morgana = self.morgana(inputs)

                continuous_mask_merlin = self.merlin.normalize_l1(continuous_mask_merlin, self.mask_size)
                continuous_mask_morgana = self.morgana.normalize_l1(continuous_mask_morgana, self.mask_size)
                
                # Convert to binary masks using top-k selection
                binary_mask_merlin = self.merlin.get_binary_mask(continuous_mask_merlin)
                binary_mask_morgana = self.morgana.get_binary_mask(continuous_mask_morgana)
            
                # Apply masks and get predictions
                masked_inputs_merlin = self.merlin.apply_mask(inputs, binary_mask_merlin)
                masked_inputs_morgana = self.morgana.apply_mask(inputs, binary_mask_morgana)
                
                logits_merlin = self.model(masked_inputs_merlin)
                logits_morgana = self.model(masked_inputs_morgana)
                
                # Calculate loss
                merlin_loss = self.merlin.criterion(logits_merlin, targets)
                morgana_loss = self.morgana.criterion(logits_morgana, targets)
                loss = merlin_loss + self.gamma * morgana_loss
                
                # Update metrics
                total_loss += loss.item()
                
                # Calculate accuracies (completeness and soundness)
                batch_completeness = get_accuracy(logits_merlin, targets, mode="merlin", idk_class=self.num_classes)
                batch_soundness = get_accuracy(logits_morgana, targets, mode="morgana", idk_class=self.num_classes)
                

                # ADD: Collect predictions for balanced accuracy
                if 'cocologic' in self.dataset_name.lower():
                    y_true_merlin.extend(targets.cpu().numpy().tolist())
                    _, predicted_merlin = logits_merlin.max(1)
                    y_pred_merlin.extend(predicted_merlin.cpu().numpy().tolist())

                    y_true_morgana.extend(targets.cpu().numpy().tolist())
                    _, predicted_morgana = logits_morgana.max(1)
                    y_pred_morgana.extend(predicted_morgana.cpu().numpy().tolist())


                # Accumulate for epoch average
                total_completeness += batch_completeness
                total_soundness += batch_soundness
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss/batch_count:.4f}',
                    'comp': f'{100.*batch_completeness:.2f}%',
                    'sound': f'{100.*batch_soundness:.2f}%'
                })
        
        # Calculate epoch-level metrics
        avg_completeness = total_completeness / batch_count if batch_count > 0 else 0
        avg_soundness = total_soundness / batch_count if batch_count > 0 else 0
                
        # Compute validation metrics
        metrics = {
            'loss': total_loss / batch_count,
            'acc': 100. * avg_completeness,  # Keep val_acc for backward compatibility
            'completeness': 100. * avg_completeness,
            'soundness': 100. * avg_soundness
        }

        # ADD: Calculate balanced accuracy for CoCoLogic
        if 'cocologic' in self.dataset_name.lower():
            metrics['val_balanced_completeness'] = 100. * balanced_accuracy_score(y_true_merlin, y_pred_merlin)
            metrics['val_balanced_soundness'] = 100. * get_balanced_soundness_cocologic(
                y_pred_morgana, y_true_morgana, self.num_classes)

        
        return metrics
    
    def extract_and_save_masks(self, output_path=None, checkpoint_path=None):
        """Extract and save Merlin and Morgana feature masks from the best model."""
        # Create default output path in same directory as embeddings
        if output_path is None:
            embeddings_dir = os.path.join(self.root_dir, f"embeddings_{self.dataset_name.lower()}")
            vocab_dir = os.path.join(embeddings_dir, f"vocab_{self.vocab_size}")
            
            if self.use_nonsparse:
                masks_dir = os.path.join(vocab_dir, "nonsparse")
            else:
                masks_dir = os.path.join(vocab_dir, f'l1_{self.l1_penalty_splice:.3f}')
                
            output_path = os.path.join(masks_dir, f'{self.dataset_name.lower()}_splice_masks_m{self.mask_size}.h5')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # If no specific checkpoint is provided, try to find the best model
        if checkpoint_path is None:
            if self.pretrained_path:
                # Use the provided pretrained_path
                checkpoint_dir = os.path.join(
                    self.pretrained_path,
                    f'{self.dataset_name}',
                    f'vocab_{self.vocab_size}',
                    f"{'nonsparse' if self.use_nonsparse else f'l1_{self.l1_penalty_splice:.3f}'}",
                    f"{self.model_name}_mask_{self.mask_size}"
                )
            else:
                # Use the default path in src/checkpoints
                checkpoint_dir = os.path.join(
                    'src', 
                    'checkpoints', 
                    f'{self.dataset_name}',
                    f'vocab_{self.vocab_size}',
                    f"{'nonsparse' if self.use_nonsparse else f'sfw_splice_l1_{self.l1_penalty_splice:.3f}'}_mask_{self.mask_size}"
                )
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        
        # Load the best model if it exists
        if os.path.exists(checkpoint_path):
            print(f"Loading best model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Restore model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore feature selector states
            if 'merlin_state_dict' in checkpoint and 'morgana_state_dict' in checkpoint:
                self.merlin.load_state_dict(checkpoint['merlin_state_dict'])
                self.morgana.load_state_dict(checkpoint['morgana_state_dict'])
                print(f"Loaded feature selectors from epoch {checkpoint['epoch']+1}")
                print(f"Expected completeness: {checkpoint.get('val_completeness', 'N/A')}%")
                print(f"Expected soundness: {checkpoint.get('val_soundness', 'N/A')}%")
                
                # Try to get learning rates
                if 'lr_merlin' in checkpoint and 'lr_morgana' in checkpoint:
                    print(f"Merlin learning rate: {checkpoint['lr_merlin']}")
                    print(f"Morgana learning rate: {checkpoint['lr_morgana']}")
                else:
                    # Use default values if not stored in checkpoint
                    print(f"Learning rates not found in checkpoint. Using default values: 0.05")
                    checkpoint['lr_merlin'] = 0.05
                    checkpoint['lr_morgana'] = 0.05
            
            # ======== VALIDATION STEP ========
            # Validate that the loaded model produces the expected metrics
            print("\nVerifying model performance matches checkpoint values...")

            print("Skipping this ... ")
            # if self.approach == "nn":
            #     val_metrics = self._validate_nn()
            # else:
            #     val_metrics = self._validate_sfw()

            # # Display actual vs expected metrics
            # print(f"Checkpoint completeness: {checkpoint.get('val_completeness', 'N/A')}%")
            # print(f"Current completeness  : {val_metrics['val_completeness']:.2f}%")
            # print(f"Checkpoint soundness  : {checkpoint.get('val_soundness', 'N/A')}%")
            # print(f"Current soundness     : {val_metrics['val_soundness']:.2f}%")
            
            # # Calculate differences
            # completeness_diff = abs(val_metrics['val_completeness'] - checkpoint.get('val_completeness', 0))
            # soundness_diff = abs(val_metrics['val_soundness'] - checkpoint.get('val_soundness', 0))
            
            # # Warning threshold (1% difference)
            # if completeness_diff > 1.0 or soundness_diff > 1.0:
            #     print(f"\n WARNING: Performance metrics differ from checkpoint values!")
            #     print(f"   Completeness difference: {completeness_diff:.2f}%")
            #     print(f"   Soundness difference  : {soundness_diff:.2f}%")
            #     print(f"   This might indicate an issue with the loaded model or feature selectors.")
            #     print(f"   Continuing with mask extraction anyway...\n")
            # else:
            #     print(f"\n✓ Model performance verified: Metrics match checkpoint values within 1%\n")
        else:
            print(f"Warning: Best model checkpoint not found at {checkpoint_path}")
            print("Using current model state for mask extraction")
        
        # Put model in eval mode
        self.model.eval()
        
        # Get original datasets (avoiding shuffled loaders)
        train_dataset = self.train_loader.dataset
        val_dataset = self.val_loader.dataset
        
        # Create non-shuffled DataLoaders specifically for extraction
        extraction_train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling to maintain consistent order
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        extraction_val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling to maintain consistent order
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Rest of the extraction code remains the same
        # Get dataset sizes
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        
        # Setup file and create datasets
        with h5py.File(output_path, 'w') as f:
            # Create datasets with chunks for efficient writing
            f.create_dataset('train_merlin_masks', shape=(train_size, self.embedding_dim),
                            dtype=np.float32, chunks=(self.batch_size, self.embedding_dim))
            f.create_dataset('train_morgana_masks', shape=(train_size, self.embedding_dim),
                            dtype=np.float32, chunks=(self.batch_size, self.embedding_dim))
            f.create_dataset('val_merlin_masks', shape=(val_size, self.embedding_dim),
                            dtype=np.float32, chunks=(self.batch_size, self.embedding_dim))
            f.create_dataset('val_morgana_masks', shape=(val_size, self.embedding_dim),
                            dtype=np.float32, chunks=(self.batch_size, self.embedding_dim))
            
            # Also save labels so analyze_masks.py can operate from a single file if desired
            f.create_dataset('train_labels', shape=(train_size,), dtype=np.int64, chunks=(min(self.batch_size, train_size),))
            f.create_dataset('test_labels', shape=(val_size,), dtype=np.int64, chunks=(min(self.batch_size, val_size),))
            
            # Store mask metadata
            metadata = f.create_group('metadata')
            metadata.attrs['mask_size'] = self.mask_size
            metadata.attrs['embedding_dim'] = self.embedding_dim
            metadata.attrs['dataset_name'] = self.dataset_name
            # metadata.attrs['l1_penalty'] = self.l1_penalty_splice
            
            # Add checkpoint info to metadata
            if os.path.exists(checkpoint_path):
                metadata.attrs['checkpoint_path'] = checkpoint_path
                metadata.attrs['checkpoint_epoch'] = checkpoint['epoch']
                metadata.attrs['val_completeness'] = checkpoint.get('val_completeness', 0)
                metadata.attrs['val_soundness'] = checkpoint.get('val_soundness', 0)
                # Add learning rates to metadata
                metadata.attrs['lr_merlin'] = checkpoint.get('lr_merlin', 0.05)  # Default to 0.05
                metadata.attrs['lr_morgana'] = checkpoint.get('lr_morgana', 0.05)  # Default to 0.05
            
            # Process training set
            print(f"Extracting masks for training set ({train_size} samples)...")
            start_idx = 0
            
            for inputs, targets in tqdm(extraction_train_loader, desc="Processing training set"):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Compute masks
                with torch.enable_grad():  # SFW needs gradients to compute masks
                    if self.approach == "nn":
                        continuous_mask_merlin = self.merlin(inputs)
                        continuous_mask_morgana = self.morgana(inputs)
                    else:
                        continuous_mask_merlin = self.merlin(inputs, targets, self.model)
                        continuous_mask_morgana = self.morgana(inputs, targets, self.model)

                    # Get binary masks
                    binary_mask_merlin = self.merlin.get_binary_mask(continuous_mask_merlin)
                    binary_mask_morgana = self.morgana.get_binary_mask(continuous_mask_morgana)
                
                # Write batch directly to file
                end_idx = start_idx + len(inputs)
                f['train_merlin_masks'][start_idx:end_idx] = binary_mask_merlin.cpu().numpy()
                f['train_morgana_masks'][start_idx:end_idx] = binary_mask_morgana.cpu().numpy()
                # Write labels as well
                f['train_labels'][start_idx:end_idx] = targets.cpu().numpy().astype('int64')
                start_idx = end_idx
                
            # Process validation set
            print(f"Extracting masks for validation set ({val_size} samples)...")
            start_idx = 0
            
            for inputs, targets in tqdm(extraction_val_loader, desc="Processing validation set"):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Compute masks
                with torch.enable_grad():
                    if self.approach == "nn":
                        continuous_mask_merlin = self.merlin(inputs)
                        continuous_mask_morgana = self.morgana(inputs)
                    else:
                        continuous_mask_merlin = self.merlin(inputs, targets, self.model)
                        continuous_mask_morgana = self.morgana(inputs, targets, self.model)
                    
                    # Get binary masks
                    binary_mask_merlin = self.merlin.get_binary_mask(continuous_mask_merlin)
                    binary_mask_morgana = self.morgana.get_binary_mask(continuous_mask_morgana)
                
                # Write batch directly to file
                end_idx = start_idx + len(inputs)
                f['val_merlin_masks'][start_idx:end_idx] = binary_mask_merlin.cpu().numpy()
                f['val_morgana_masks'][start_idx:end_idx] = binary_mask_morgana.cpu().numpy()
                # Write validation labels as test_labels for compatibility with analyze_masks
                f['test_labels'][start_idx:end_idx] = targets.cpu().numpy().astype('int64')
                start_idx = end_idx
        
        print(f"Masks successfully saved to {output_path}")
        return output_path

    def _load_saved_masks(self):
        """Load saved masks for reconstruction regularization"""
        if not self.feature_selector_config.use_saved_masks:
            return
        
        masks_path = self.feature_selector_config.saved_masks_path
        if masks_path is None:
            # Try to find masks automatically using the proper directory structure
            embeddings_dir = os.path.join(self.root_dir, f"embeddings_{self.dataset_name.lower()}")
            
            # Add vocabulary size to path structure
            vocab_dir = os.path.join(embeddings_dir, f"vocab_{self.vocab_size}")
            
            # Use nonsparse or sparse directory based on configuration
            if self.use_nonsparse:
                masks_dir = os.path.join(vocab_dir, "nonsparse")
            else:
                masks_dir = os.path.join(vocab_dir, f'l1_{self.l1_penalty_splice:.3f}')
                
            # Construct the complete path using the mask size
            masks_path = os.path.join(masks_dir, f'{self.dataset_name.lower()}_splice_masks_m{self.mask_size}.h5')
        
        if not os.path.exists(masks_path):
            print(f"Warning: Saved masks not found at {masks_path}. Reconstruction regularization will be disabled.")
            self.feature_selector_config.use_saved_masks = False
            return
        
        print(f"Loading saved masks from {masks_path}")
        try:
            with h5py.File(masks_path, 'r') as f:
                # Load train masks
                self.saved_train_merlin_masks = torch.from_numpy(f['train_merlin_masks'][:]).float()
                self.saved_train_morgana_masks = torch.from_numpy(f['train_morgana_masks'][:]).float()
                
                # Load validation masks as well
                if 'val_merlin_masks' in f and 'val_morgana_masks' in f:
                    self.saved_val_merlin_masks = torch.from_numpy(f['val_merlin_masks'][:]).float()
                    self.saved_val_morgana_masks = torch.from_numpy(f['val_morgana_masks'][:]).float()
                    print(f"Loaded {len(self.saved_val_merlin_masks)} validation masks")
                else:
                    print("No validation masks found in saved file")
                    self.saved_val_merlin_masks = None
                    self.saved_val_morgana_masks = None
                
                # Print mask stats
                print(f"Loaded {len(self.saved_train_merlin_masks)} train masks")
                print(f"Merlin mask avg features: {self.saved_train_merlin_masks.sum(1).mean().item():.1f}")
                print(f"Morgana mask avg features: {self.saved_train_morgana_masks.sum(1).mean().item():.1f}")
                
                # Load metadata
                checkpoint_info = f['metadata'].attrs.get('checkpoint_path', 'Unknown')
                completeness = f['metadata'].attrs.get('val_completeness', 0)
                soundness = f['metadata'].attrs.get('val_soundness', 0)
                print(f"Masks from checkpoint: {os.path.basename(checkpoint_info)}")
                print(f"- Completeness: {completeness:.2f}%")
                print(f"- Soundness: {soundness:.2f}%")
                
            print("✓ Successfully loaded saved masks for reconstruction regularization")
            
            # Create both training and validation index mappings
            if self.feature_selector_config.use_saved_masks:
                self._create_train_indices_mapping()
                self._create_val_indices_mapping()
        
        except Exception as e:
            print(f"Error loading saved masks: {e}")
            print("Disabling reconstruction regularization")
            self.feature_selector_config.use_saved_masks = False

    def save_checkpoint(self, state: Dict, filename: str):
        """Save model checkpoint"""
        torch.save(state, filename)

