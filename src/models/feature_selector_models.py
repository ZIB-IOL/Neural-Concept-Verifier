import torch.nn as nn
from models.classifier import SAB
import torch
import torch.nn.functional as F

class MLPFeatureSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        
        self.output_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.output_dim),
        )

    def forward(self, x):
        continuous_mask = self.network(x)
        return continuous_mask # sigmoid degrades performance


class DeepResidualMLPFeatureSelector(nn.Module):
    """
    A deep MLP with 10 hidden layers and residual connections for better gradient flow.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        
        self.output_dim = input_dim
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Create 10 hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        for _ in range(10):
            block = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.hidden_layers.append(block)
            
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)
        
    def forward(self, x):
        # Initial projection
        h = self.input_proj(x)
        
        # Apply hidden layers with residual connections
        for layer in self.hidden_layers:
            h_new = layer(h)
            h = h + h_new  # Skip connection
            
        # Output projection
        continuous_mask = self.output_proj(h)
        return continuous_mask
    
    
class SetTransformerFeatureSelector(nn.Module):
    def __init__(self, input_dim: int, num_slots: int, num_blocks: int, dim_hidden: int = 128, num_heads: int = 4, ln: bool = True, dropout: float = 0.1):
        super(SetTransformerFeatureSelector, self).__init__()
        
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.num_blocks = num_blocks
        
        # Set Transformer blocks
        self.enc = nn.Sequential(
            SAB(dim_in=input_dim, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
            nn.Dropout(dropout),
            SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
        )
        
        # Generate mask for each slot - MODIFIED to output with dimension matching input_dim
        self.mask_generator = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, input_dim)  # Changed from num_blocks to input_dim
        )

    def forward(self, x):
        # Add a sequence dimension if the input is just [batch_size, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Now [batch_size, 1, features]
        
        x = self.enc(x)  # Process through transformer blocks
        
        # Since we have a sequence dim now, we need to squeeze it back
        x = x.squeeze(1)  # Back to [batch_size, dim_hidden]
        
        # Generate mask scores - output dimension is now [batch_size, input_dim]
        mask = self.mask_generator(x)
        return mask # sigmoid degrades the performance of the model
    

class SparseAwareMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        
        self.output_dim = input_dim
        
        # Initial sparse feature processing with smaller hidden dims
        self.sparse_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Normalize to handle varying sparsity levels
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature importance scoring with bottleneck
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, input_dim)
        )
        
    def forward(self, x):
        # Count non-zero features per sample for adaptive processing
        # non_zero_counts = (x.abs() > 1e-6).sum(1).float().unsqueeze(1)
        
        # Process through network
        features = self.sparse_encoder(x)
        
        # Use sparsity information to adapt the features
        # sparsity_factor = torch.clamp(non_zero_counts / 100.0, 0.1, 1.0)
        # features = features * sparsity_factor

        # Generate mask scores, conditioned on sparsity level
        scores = self.scorer(features)
        
        return scores


class SparseFeatureAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Initial feature embedding
        self.feature_embed = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Final projection back to input dimension
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Embed features
        h = self.feature_embed(x)
        
        # Generate attention scores - focus on non-zero inputs
        non_zero_mask = (x.abs() > 1e-6).float()
        attention = self.attention(h).sigmoid() * non_zero_mask
        
        # Apply attention and project back to input dimensions
        attended = h * attention
        scores = self.output_proj(attended)
        
        return scores


class SimpleMLPFeatureSelector(nn.Module):
    """
    A minimal MLP feature selector with just one hidden layer.
    Much fewer parameters than the other models.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.output_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x):
        continuous_mask = self.network(x)
        return continuous_mask # sigmoid degrades the performance of the model


class NarrowMLPFeatureSelector(nn.Module):
    """
    A narrow but slightly deeper MLP with very few parameters.
    Uses smaller hidden dimensions to prevent overfitting.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        
        self.output_dim = input_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Help stabilize training with fewer parameters
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x):
        continuous_mask = self.network(x)
        return continuous_mask


class LightweightTransformerSelector(nn.Module):
    def __init__(self, input_dim, num_heads=4, dim_feedforward=128, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        
    def forward(self, x):
        # Add batch dimension if needed for single vector
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Self attention block
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2)[0]
        
        # Feed forward block
        x2 = self.norm2(x)
        x = x + self.dropout(self.linear2(torch.relu(self.linear1(x2))))
        
        if squeeze_output:
            x = x.squeeze(1)
        return x


class SmallLightweightTransformerSelector(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=4, dim_feedforward=128, dropout=0.0):
        """
        Transformer block with input/output projections to handle large input_dim.
        
        Args:
            input_dim (int): Dimension of the input concept vector (e.g., 10000).
            embed_dim (int): Internal dimension for the transformer block. Much smaller than input_dim.
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        # Project input down to a smaller embedding dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Transformer components operating on the smaller embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,  # Use smaller dimension here
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        # Project output back to the original input dimension
        self.output_proj = nn.Linear(embed_dim, input_dim)
        
    def forward(self, x):
        # Project input
        x = self.input_proj(x)
        
        # --- Transformer Block ---
        # Self attention block
        x_res = x # Store residual connection
        x2 = self.norm1(x)
        attn_output, _ = self.self_attn(x2, x2, x2)
        x = x_res + attn_output # Add residual
        
        # Feed forward block
        x_res = x # Store residual connection
        x2 = self.norm2(x)
        ff_output = self.linear2(torch.relu(self.linear1(x2)))
        x = x_res + self.dropout(ff_output) # Add residual
        # --- End Transformer Block ---

        # Project output back to input dimension
        output_scores = self.output_proj(x)
        
        return output_scores

class GatedFeatureSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Create feature context representation
        context = self.context_encoder(x)
        
        # Generate gates - how important each feature is
        gates = self.gate_network(x)
        
        # Apply gates to get feature scores
        return gates * context


class LinearFeatureSelector(nn.Module):
    """
    The simplest possible feature selector - just a single linear layer.
    Very few parameters, minimal capacity.
    """
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.output_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.linear(x)


class LowRankLinearFeatureSelector(nn.Module):
    """
    Low-rank approximation of a linear layer for extreme parameter efficiency.
    Approximates a full input_dim x input_dim matrix with two smaller matrices.
    """
    def __init__(self, input_dim: int, rank: int = 64, **kwargs):
        super().__init__()
        self.output_dim = input_dim
        self.rank = rank
        
        # Factorize into two low-rank matrices
        self.down_proj = nn.Linear(input_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, input_dim, bias=True)
        
    def forward(self, x):
        return self.up_proj(self.down_proj(x))


class ResidualFeatureSelector(nn.Module):
    """
    A small network with skip connections for better gradient flow.
    Uses fewer layers but maintains expressiveness through residual connections.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.output_dim = input_dim
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.residual_block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.residual_block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Initial projection
        h = self.input_proj(x)
        
        # First residual block
        h = h + self.residual_block1(h)
        
        # Second residual block
        h = h + self.residual_block2(h)
        
        # Output projection
        continuous_mask = self.output_proj(h)
        return continuous_mask


class BottleneckFeatureSelector(nn.Module):
    """
    Uses a bottleneck architecture with a very narrow middle layer.
    Forces the model to learn a compressed representation of the features.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, bottleneck_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.output_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),  # Aggressive bottleneck
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        bottleneck = self.encoder(x)
        continuous_mask = self.decoder(bottleneck)
        return continuous_mask


class DenseNetFeatureSelector(nn.Module):
    """
    Inspired by DenseNet architecture with dense connections between layers.
    Each layer receives features from all preceding layers for better information flow.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 4, growth_rate: int = 16, dropout: float = 0.1):
        super().__init__()
        self.output_dim = input_dim
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Dense blocks
        self.layers = nn.ModuleList()
        current_dim = hidden_dim
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.LayerNorm(current_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(current_dim, growth_rate)
            )
            self.layers.append(layer)
            current_dim += growth_rate
        
        # Final output projection
        self.output_proj = nn.Linear(current_dim, input_dim)
        
    def forward(self, x):
        features = [self.input_proj(x)]
        
        for layer in self.layers:
            # Concatenate all previous features
            h = torch.cat(features, dim=1)
            new_features = layer(h)
            features.append(new_features)
        
        # Concatenate final representation
        h = torch.cat(features, dim=1)
        
        # Output projection
        continuous_mask = self.output_proj(h)
        return continuous_mask


class GatedResidualNetwork(nn.Module):
    """
    Combines residual connections with gating mechanisms for adaptive feature selection.
    Gates control how much of each residual connection passes through.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.output_dim = input_dim
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Gated residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            # Main path
            main_path = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # Gate network
            gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            
            self.blocks.append(nn.ModuleDict({
                'main': main_path,
                'gate': gate
            }))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Initial projection
        h = self.input_proj(x)
        
        # Gated residual blocks
        for block in self.blocks:
            main_output = block['main'](h)
            gate_values = block['gate'](h)
            h = h + gate_values * main_output
        
        # Output projection
        continuous_mask = self.output_proj(h)
        return continuous_mask
    



# Used Model for U-Net
class SimpleNet(nn.Module):
    def __init__(self, n_channels: int, bilinear: bool = True, apply_sigmoid: bool = False):
        super(SimpleNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.apply_sigmoid = apply_sigmoid

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.out_conv = OutConv(64, 1)
        self.lin = nn.Linear(32 * 32, 784)

    def forward(self, x):
        x1 = self.inc(x)  # Output shape: Channel=64, Width=28, Height=28
        x2 = self.down1(x1)  # Channel=128, Width=14, Height=14
        x3 = self.down2(x2)  # Channel=256, Width=7, Height=7
        x4 = self.down3(x3)  # Channel=512, Width=3, Height=3
        x5 = self.down4(x4)  # Channel=512, Width=1, Height=1
        x = self.up1(x5, x4)  # Channel=256, Width=3, Height=3
        x = self.up2(x, x3)  # Channel=128, Width=7, Height=7
        x = self.up3(x, x2)  # Channel=64, Width=14, Height=14
        x = self.up4(x, x1)  # Channel=128, Width=28, Height=28
        logits = self.out_conv(x)  # N, C, W, H
        # a = torch.abs(logits[:, 0, :, :])
        # b = torch.abs(logits[:, 1, :, :])
        # logits = torch.unsqueeze(a / (a + b), dim=1)
        # return shape: N, C, W, H
        # return torch.sigmoid(logits)
        return torch.sigmoid(logits) if self.apply_sigmoid else logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)