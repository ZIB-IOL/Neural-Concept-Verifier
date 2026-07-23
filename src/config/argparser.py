import argparse


# fmt: off
def add_trainer_args(parser):
    # Keep existing trainer arguments
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--approach", type=str, required=True, choices=["regular", "sfw", "nn", "posthoc", "unet"], 
                      help="Training approach to use")
    parser.add_argument("-d", "--debug", action=argparse.BooleanOptionalAction, default=False, 
                      help="Enable debug mode with reduced dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, 
                      help="Enable wandb logging")
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True,
                       help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10,
                       help="Patience for early stopping")
    parser.add_argument("--soundness_threshold", type=float, default=0.75,
                      help="Threshold for soundness to consider a model good")
    
    # Add new mask extraction arguments
    parser.add_argument("--extract_masks", action=argparse.BooleanOptionalAction, default=False,
                      help="Extract and save masks after training (only for SFW approach)")
    parser.add_argument("--masks_output_path", type=str, default=None,
                      help="Path to save extracted masks (default: same directory as embeddings)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                      help="Path to specific checkpoint to load for mask extraction")
    parser.add_argument("--skip_training", action=argparse.BooleanOptionalAction, default=False,
                      help="Skip training and only extract masks from existing checkpoint")
    # Analysis / precision+entropy computation flags
    parser.add_argument("--compute_precision_entropy", action=argparse.BooleanOptionalAction, default=False,
                      help="Compute average precision and conditional entropy from extracted masks")
    parser.add_argument("--compute_only", action=argparse.BooleanOptionalAction, default=False,
                      help="If set with --compute_precision_entropy, run only that computation and exit")
    parser.add_argument("--selector", type=str, choices=["merlin", "morgana"], default="merlin",
                      help="Which selector's masks to use for precision/entropy computation")
    parser.add_argument("--tolerance", type=int, default=2,
                      help="Hamming-distance tolerance to count a mask match")
    parser.add_argument("--target_class", type=int, default=None,
                      help="Optional: compute metrics only for this class index")
    parser.add_argument("--analysis_seed", type=int, default=42,
                      help="RNG seed used for inner shuffling in precision/entropy computation")

def add_dataset_args(parser):
    dataset_args = parser.add_argument_group('Dataset args')
    dataset_args.add_argument("--root_dir", type=str, help="Root directory for datasets", default="./data")
    dataset_args.add_argument("--dataset", type=str, help="Dataset", required=True)
    dataset_args.add_argument("--l1_penalty_splice", type=float, help="L1 penalty value to use for loading embeddings", default=0.10)
    dataset_args.add_argument("--batch_size", type=int, help="Batch Size", required=True)
    dataset_args.add_argument("--add_normalization", action=argparse.BooleanOptionalAction, help="Normalizes the input images")
    dataset_args.add_argument("--num_workers", type=int, help="Number of workers for data loading", default=4)
    dataset_args.add_argument("--vocab_size", type=int, help="Vocabulary size for embeddings", default=10000)
    dataset_args.add_argument("--use_nonsparse", action=argparse.BooleanOptionalAction, 
                             help="Use nonsparse embeddings instead of L1-regularized ones")
    dataset_args.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, help="Use class weights for loss function", default=True)

def add_boolean_args(parser):
    boolean_args = parser.add_argument_group('BooleanOptionalAction args')
    boolean_args.add_argument("--save_model", action=argparse.BooleanOptionalAction, help="Save model")
    boolean_args.add_argument("--use_amp", action=argparse.BooleanOptionalAction, help="Use automatic mixed precision")
    boolean_args.add_argument("--cuda_benchmark", action=argparse.BooleanOptionalAction, help="Use cuda benchmark (not recommended)")

def add_model_args(parser):
    model_args = parser.add_argument_group('model specific args')
    model_args.add_argument("--model", type=str, help="Model for Arthur, e.g., SimpleCNN, DeeperCNN, ResNet18 etc.")
    model_args.add_argument("--imagenet_pretrained", action=argparse.BooleanOptionalAction, help="Use pretrained model for Arthur's regular training")
    model_args.add_argument("--lr", type=float, help="Learning Rate of Arthur", default=1e-4)
    model_args.add_argument("--weight_decay", type=float, help="Weight Decay for Arthur", default=0)
    model_args.add_argument("--hidden_dim", type=int, help="Hidden Dimension for MLP Classifier", default=512)
    model_args.add_argument("--dropout", type=float, help="Dropout for MLP Classifier", default=0.3)
    model_args.add_argument("--low_rank", type=int, help="Rank for Low Rank Linear Classifier")
    model_args.add_argument("--pretrained_model", action=argparse.BooleanOptionalAction, help="Use pretrained model")
    model_args.add_argument("--pretrained_path", type=str, help="Path to pretrained model")
    model_args.add_argument("--load_sfw_model", action=argparse.BooleanOptionalAction, default=False,
                      help="Force loading SFW model instead of regular model")
    # Learning rate scheduler arguments
    model_args.add_argument("--use_lr_scheduler", action=argparse.BooleanOptionalAction, help="Use learning rate scheduler")
    model_args.add_argument("--lr_scheduler_type", type=str, choices=["plateau", "step", "cosine", "exponential"], 
                          default="plateau", help="Type of learning rate scheduler")
    model_args.add_argument("--lr_scheduler_patience", type=int, default=5, help="Patience for plateau scheduler")
    model_args.add_argument("--lr_scheduler_factor", type=float, default=0.1, help="Factor to reduce LR by")
    model_args.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6, help="Minimum learning rate")
    model_args.add_argument("--lr_scheduler_step_size", type=int, default=10, help="Step size for step scheduler")
    model_args.add_argument("--lr_scheduler_gamma", type=float, default=0.1, help="Gamma for step and exponential schedulers")


def add_feature_selector_args(parser):
    feature_selector_args = parser.add_argument_group('General Feature Selector (SFW or NN) args')
    feature_selector_args.add_argument("--feature_selector_architecture", type=str, help="Feature Selector Architecture (e.g., `mlp`, `settransformer`)")
    feature_selector_args.add_argument("--segmentation_method", type=str, help="Segmentation method for Merlin and Morgana (only topk atm)")
    feature_selector_args.add_argument("--mask_size", type=int, help="Size of Mask")
    feature_selector_args.add_argument("--lr_merlin", type=float, help="Learning Rate of Merlin either as NN or SFW Optimizer")
    feature_selector_args.add_argument("--lr_morgana", type=float, help="Learning Rate of Morgana either as NN or SFW Optimizer")
    feature_selector_args.add_argument("--gamma", type=float, help="Gamma for weighting the loss between Merlin and Morgana")
    feature_selector_args.add_argument("--l1_penalty_coefficient", type=float, help="L1 penalty coefficient for SFW")
    feature_selector_args.add_argument("--overlap_weight", type=float, help="Weight for overlapping features")
    feature_selector_args.add_argument("--weight_decay_merlin", type=float, help="Weight decay for Neural Feature Selector (Merlin)", default=0)
    feature_selector_args.add_argument("--weight_decay_morgana", type=float, help="Weight decay for Neural Feature Selector (Morgana)", default=0)
    feature_selector_args.add_argument("--track_mask_statistics", action=argparse.BooleanOptionalAction, help="Track mask overlap and feature inclusion statistics")
    feature_selector_args.add_argument("--feature_inclusion_weight", type=float, help="Weight for feature inclusion regularization", default=10.0)
    feature_selector_args.add_argument("--prioritize_nonzero", action=argparse.BooleanOptionalAction, help="Prioritize non-zero features in mask selection", default=False)
    # Add new arguments for mask reconstruction
    feature_selector_args.add_argument("--use_saved_masks", action=argparse.BooleanOptionalAction, default=False, help="Use saved masks as reconstruction targets")
    feature_selector_args.add_argument("--saved_masks_path", type=str, default=None, help="Path to saved masks file (h5 format)")
    feature_selector_args.add_argument("--mask_reconstruction_weight", type=float, default=0.1, help="Weight for mask reconstruction loss")
    feature_selector_args.add_argument("--use_continuous_reconstruction", action=argparse.BooleanOptionalAction, default=False, help="Use continuous mask for reconstruction, otherwise Straight-Through-Estimator (STE) is used")
    feature_selector_args.add_argument("--weight_positive_examples", action=argparse.BooleanOptionalAction, default=False,
                      help="Apply weighting to positive examples in BCE loss to address class imbalance. Seems to help a lot!")