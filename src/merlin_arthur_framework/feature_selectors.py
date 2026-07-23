import torch
from merlin_arthur_framework.stochastic_frank_wolfe import SFW, PositiveKSparsePolytope
from typing import Optional


class SFWFeatureSelector(torch.nn.Module):
    def __init__(
        self,
        mask_size: int,
        mode: str = "merlin",
        lr_merlin: float = 0.1,
        lr_morgana: float = 0.1,
        binary_classification: bool = False,
        l1_penalty_coefficient: float = 0.01,
        overlap_weight: float = 0.0,
        idk_class: int = 100
    ) -> None:
        """
        Simple feature selector using Stochastic Frank-Wolfe optimization.
        
        Args:
            mask_size: Number of features to select (sparsity level)
            mode: "merlin" (minimize loss) or "morgana" (maximize loss)
            lr_merlin: Learning rate (or step size) for Merlin (SFW)
            lr_morgana: Learning rate (or step size) for Morgana (SFW)
            binary_classification: Whether this is a binary classification task
            l1_penalty_coefficient: Coefficient for L1 regularization
            overlap_weight: Weight for overlapping features
            idk_class: Index of the IDK class for Morgana
        """
        super().__init__()
        assert mode in ["merlin", "morgana"], "Mode must be 'merlin' or 'morgana'"
        assert mask_size > 0, "Mask size must be greater than 0"
        
        self.mask_size = mask_size
        self.mode = mode
        self.lr_merlin = lr_merlin
        self.lr_morgana = lr_morgana
        self.binary_classification = binary_classification
        self.l1_penalty_coefficient = l1_penalty_coefficient
        self.overlap_weight = overlap_weight
        self.idk_class = idk_class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set criterion based on classification type
        if binary_classification:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss() if mode == "merlin" else MorganaCriterion(self.idk_class)
            
    def forward(self, x, y, classifier, init_mask=None):
        """
        Optimize a mask using SFW for embedding vectors.
        
        Args:
            x: Input embedding tensor [batch_size, embedding_dim]
            y: Target labels [batch_size]
            classifier: Model to use for predictions
            init_mask: Initial mask (random if None)
            
        Returns:
            Optimized mask
        """
        batch_size, embedding_dim = x.shape
        
        # Initialize mask if needed
        if init_mask is None:
            init_mask = torch.rand_like(x)  # Full embedding dimension
            # # Start with values proportional to input magnitudes
            # init_mask = torch.abs(x.detach().clone())
            # # Normalize
            # init_mask = init_mask / (init_mask.max() + 1e-10)
            
        # Setup SFW optimizer
        constraint = PositiveKSparsePolytope(
            n=embedding_dim,  # Number of features in embedding
            bs=batch_size, 
            k=self.mask_size
        )
        mask = constraint.shift_inside(init_mask)
        # Create parameter for optimization
        mask = torch.nn.Parameter(mask.to(self.device), requires_grad=True)
        # Use configurable learning rate from class
        optimizer = SFW([mask], learning_rate=self.lr_merlin, momentum=0.9)
        
        # Freeze classifier parameters
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        
        # Track best loss for early stopping
        best_loss = float('inf')
        patience = 8
        patience_counter = 0
        
        # Optimization loop
        for iteration in range(250):  # Max iterations
            optimizer.zero_grad()
            
            # Apply mask and get predictions
            x_masked = self.apply_mask(x, mask)
            logits = classifier(x_masked)
            
            # Handle binary classification case
            if self.binary_classification:
                logits = logits.squeeze(1)
                y_tensor = y.float()
            else:
                y_tensor = y
            
            # Calculate loss based on mode
            if self.mode == "merlin":
                distortion = self.criterion(logits, y_tensor)
            elif self.mode == "morgana": 
                distortion = -self.criterion(logits, y_tensor)
                
            if self.l1_penalty_coefficient is not None:
                # Add regularization
                l1_penalty = self.l1_penalty_coefficient * torch.mean(torch.abs(mask))
            else:
                l1_penalty = 0.0
            
            # Total loss
            loss = distortion + l1_penalty
            loss.backward()
            
            # Update mask
            optimizer.step(constraints=[constraint])
            
            # Early stopping check
            if loss.item() < best_loss - 1e-5:  # Improved by at least delta
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Re-enable classifier gradients
        for param in classifier.parameters():
            param.requires_grad = True
        
        return mask.detach()
    
    def apply_mask(self, x, mask):
        """Apply mask to input"""
        return mask * x

    def get_binary_mask(self, continuous_mask):
        """Convert continuous mask to binary mask"""
        v = torch.zeros_like(continuous_mask).flatten(start_dim=1)
        max_indices = torch.topk(torch.abs(continuous_mask.flatten(start_dim=1)), k=self.mask_size).indices.to(continuous_mask.device)
        v.scatter_(1, max_indices, 1.0)
        return v.reshape(continuous_mask.shape)
    

class NeuralFeatureSelector(torch.nn.Module):
    def __init__(
        self,
        mask_size: int,
        lr = 0.01,
        model: torch.nn.Module = None,
        mode: str = "merlin",
        idk_class: int = 100,
        binary_classification: bool = False,
        prioritize_nonzero: bool = False
    ) -> None:
        super().__init__()

        assert mode in ["merlin", "morgana"], "Mode must be 'merlin' or 'morgana'"
        assert mask_size > 0, "Mask size must be greater than 0"

        self.lr = lr
        self.mask_size = mask_size
        self.mode = mode
        self.idk_class = idk_class
        self.model = model
        self.prioritize_nonzero = prioritize_nonzero

        # Set criterion based on classification type
        if binary_classification:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss() if mode == "merlin" else MorganaCriterion(self.idk_class)

    def forward(self, x):
        # Store reference to inputs for prioritization
        if self.prioritize_nonzero:
            self.last_inputs = x
        
        # Forward pass through the model
        return self.model(x)
    
    def apply_mask(self, x, mask):
        """Apply mask to input"""
        return mask * x

    def get_binary_mask(self, continuous_mask):
        # pick top-k by absolute value along feature-dim
        _, topk_indices = torch.topk(
            torch.abs(continuous_mask),
            k=self.mask_size,
            dim=1
        )
        binary_mask = torch.zeros_like(continuous_mask)
        binary_mask.scatter_(1, topk_indices, 1.0)
        return binary_mask

    def normalize_l1(self, mask, mask_size):
        """
        Normalize mask so its L1 norm equals mask_size, unless already smaller.
        For concept embeddings with shape [batch_size, features].
        """
        # Calculate L1 norm along feature dimension
        l1_norm = torch.norm(mask, p=1, dim=1, keepdim=True)
        
        # Compute scaling factor, capped at 1.0
        factor = torch.clamp(mask_size / (l1_norm + 1e-7), max=1.0)
        
        # Apply scaling
        normalized_mask = factor * mask
        return normalized_mask
    
# U-Net
class PixelFeatureSelector(torch.nn.Module):
    def __init__(
        self,
        mask_size: int,
        mode: str = "merlin",
        idk_class: int = 3,
        model: torch.nn.Module = None
    ) -> None:
        super().__init__()

        assert mode in ["merlin", "morgana"], "Mode must be 'merlin' or 'morgana'"
        assert mask_size > 0, "Mask size must be greater than 0"

        self.mask_size = mask_size
        self.mode = mode
        self.idk_class = idk_class
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss() if mode == "merlin" else MorganaCriterion(self.idk_class)

    def forward(self, x):
        return self.model(x)
    
    def apply_mask(self, x, mask):
        """Apply mask to input"""
        # Add gaussian noise to masked input
        x_masked = mask * x + (1 - mask) * torch.rand_like(x)
        return x_masked
        
    def get_binary_mask(self, continuous_mask):
        """Convert continuous mask to binary mask"""
        v = torch.zeros_like(continuous_mask).flatten(start_dim=1)
        max_indices = torch.topk(torch.abs(continuous_mask.flatten(start_dim=1)), k=self.mask_size).indices.to(continuous_mask.device)
        v.scatter_(1, max_indices, 1.0)
        return v.reshape(continuous_mask.shape) 
    
    def normalize_l1(self, input: torch.Tensor, mask_size: int):
        factor = torch.clamp(mask_size / (1e-7 + torch.norm(input, p=1, dim=(2, 3), keepdim=True)), max=1)  # type: ignore
        return factor * input


class MorganaCriterion(torch.nn.Module):
    def __init__(self, idk_class, weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.idk_class = idk_class # index of the idk class

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns the loss that is minimized by Arthur and maximized by Morgana.

        Args:
            logits (torch.Tensor): Arthurs output.
            target (torch.Tensor): True targets.

        Raises:
            ValueError: Reduction assertion, possible values are `mean`, `sum` and `none`.

        Returns:
            torch.Tensor: Outputs loss minimized by Arthur and maximized by Morgana.
        """
        logits_wrt_true_class = torch.gather(logits, dim=1, index=target.unsqueeze(1))
        logits_idk = logits[:, self.idk_class].unsqueeze(1)  # last column corresponds to idk logits
        logits_concatenated = torch.cat((logits_wrt_true_class, logits_idk), 1)

        diff = -torch.abs(logits_wrt_true_class - logits_idk)

        target_cloned = torch.clone(target)
        target_cloned[torch.argmax(logits_concatenated, dim=1) == 1] = self.idk_class
        criterion = torch.nn.CrossEntropyLoss(weight=self.weight, reduction=self.reduction)

        if self.reduction == "mean":
            correction_term = -torch.log(1 + torch.exp(diff)).mean()
        elif self.reduction == "sum":
            correction_term = -torch.log(1 + torch.exp(diff)).sum()
        elif self.reduction == "none":
            correction_term = -torch.log(1 + torch.exp(diff)).squeeze()
        else:
            raise ValueError(f"unexpected value for reduction, got `{self.reduction}`")

        loss = criterion(logits, target_cloned) + correction_term

        return loss