try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
from typing import Dict, Any, Optional
import torch
from dataclasses import asdict, is_dataclass

def dataclass_to_dict(dataclass_instance):
    return asdict(dataclass_instance) if is_dataclass(dataclass_instance) else dataclass_instance

def initialize_wandb(config: Dict[str, Any]):
    """Initializes the logger.

    Args:
        config (Dict[str, Any]): Dictionary containing all hyperparameters and configurations.
    Returns:
        logger (wandb or None): Weights and Biases Logger.
    """
    logger = None
    if config["trainer_config"].wandb is True:
        if not WANDB_AVAILABLE:
            print("Warning: --wandb set but wandb is not installed; skipping W&B logging.")
            return None
        config_dict_converted = {
            k: (dataclass_to_dict(v) if hasattr(v, '__dataclass_fields__') else v)
            for k, v in config.items()
        }
        wandb.init(project="Concept-Learning", config=config_dict_converted)
        logger = wandb
    return logger

@torch.no_grad()
def get_accuracy(logits: torch.Tensor, y_true: torch.Tensor, mode: str, idk_class: int, binary_classification: Optional[bool] = None
) -> float:
    if binary_classification is True:
        prediction = torch.round(torch.sigmoid(logits)).squeeze()  # convert probabilities to binary predictions
    else:
        prediction = torch.argmax(logits, dim=1)
    if mode == "merlin":
        accuracy = prediction.eq(y_true.squeeze()).sum().item() / float(len(y_true))
    elif mode == "morgana" and binary_classification is not True:
        accuracy = torch.logical_or(
            prediction.eq(y_true.squeeze()), 
            prediction.eq(idk_class)
        ).sum().item() / float(len(y_true))
    elif mode == "morgana" and binary_classification is True:
        accuracy = prediction.eq(y_true.squeeze()).sum().item() / float(len(y_true))
    else:
        raise ValueError(f"Unexpected value for mode, got `{mode}`")
    return accuracy

# def get_balanced_accuracy(logits: torch.Tensor, y_true: torch.Tensor, mode: str, idk_class: int) -> float:
#     """Calculate balanced accuracy for CoCoLogic datasets only"""
#     from sklearn.metrics import balanced_accuracy_score
    
#     prediction = torch.argmax(logits, dim=1)
    
#     if mode == "merlin":
#         # Only calculate for known classes (exclude idk_class)
#         valid_mask = (y_true != idk_class)
#         if valid_mask.sum() > 1:  # Need at least 2 samples
#             y_true_filtered = y_true[valid_mask].cpu().numpy()
#             y_pred_filtered = prediction[valid_mask].cpu().numpy()
#             return balanced_accuracy_score(y_true_filtered, y_pred_filtered)
#         else:
#             return 0.0
#     elif mode == "morgana":
#         # For morgana, we could calculate balanced accuracy for the binary problem
#         # (correct/idk vs incorrect), but this might not be as meaningful
#         # For now, return regular accuracy
#         return get_accuracy(logits, y_true, mode, idk_class)
#     else:
#         raise ValueError(f"Unexpected value for mode, got `{mode}`")
    
def get_balanced_soundness_cocologic(y_pred_list: list, y_true_list: list, idk_class: int) -> float:
    """Calculate balanced soundness for COCOLogic datasets from collected predictions"""
    import numpy as np
    
    y_true_np = np.array(y_true_list)
    y_pred_np = np.array(y_pred_list)
    
    # For each class, calculate soundness (fraction of samples where Morgana was sound)
    unique_classes = np.unique(y_true_np)
    class_soundness = []
    
    for cls in unique_classes:
        # Get all samples of this class
        class_mask = (y_true_np == cls)
        class_predictions = y_pred_np[class_mask]
        
        # Count sound predictions (correct class or IDK)
        sound_predictions = np.logical_or(
            class_predictions == cls,  # Predicted correct class
            class_predictions == idk_class  # Predicted IDK
        )
        
        # Calculate soundness for this class
        class_soundness_rate = np.mean(sound_predictions) if len(class_predictions) > 0 else 0
        class_soundness.append(class_soundness_rate)
    
    # Return balanced average across all classes
    return np.mean(class_soundness)