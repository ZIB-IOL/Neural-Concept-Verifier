import torch.nn as nn
import torch
import math
from torchvision import models
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    """Basic linear classifier"""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)
    

# low rank linear classifier
class LowRankLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, rank=32):
        super().__init__()
        self.linear = nn.Linear(input_dim, rank, bias=False)
        self.rank_linear = nn.Linear(rank, num_classes)
        
    def forward(self, x):
        x = self.linear(x)
        return self.rank_linear(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class ShallowMLPRelu(nn.Module):
    def __init__(self, input_dim=10_000, hidden_dim=100, num_classes=10, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 10_000 × 100 = 1,000,000 params
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # 100 × 10 = 1,000
        )

    def forward(self, x):
        return self.net(x)


class ShallowMLPGelu(nn.Module):
    def __init__(self, input_dim=10_000, hidden_dim=100, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 10_000 × 100 = 1,000,000 params
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # 100 × 10 = 1,000
        )

    def forward(self, x):
        return self.net(x)



class NonLinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class ShallowMLPRelu(nn.Module):
    def __init__(self, input_dim=10_000, hidden_dim=100, num_classes=10, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 10_000 × 100 = 1,000,000 params
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # 100 × 10 = 1,000
        )

    def forward(self, x):
        return self.net(x)


class ShallowMLPGelu(nn.Module):
    def __init__(self, input_dim=10_000, hidden_dim=100, num_classes=10, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 10_000 × 100 = 1,000,000 params
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # 100 × 10 = 1,000
        )

    def forward(self, x):
        return self.net(x)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    """
    Set Transformer used classification on encodings.
    """
    def __init__(self, dim_input=3, dim_output=40, dim_hidden=128, num_heads=4, ln=False):
        """
        Builds the Set Transformer.
        :param dim_input: Integer, input dimensions
        :param dim_output: Integer, output dimensions
        :param dim_hidden: Integer, hidden layer dimensions
        :param num_heads: Integer, number of attention heads
        :param ln: Boolean, whether to use Layer Norm
        """
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=dim_input, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
            nn.Dropout(),
            SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim=dim_hidden, num_heads=num_heads, num_seeds=1, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze()
    

# ResNet18 on pixel space
class ResNet18(nn.Module):
    def __init__(self, dim_output: int):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, dim_output)
        
    def forward(self, x):
        return self.resnet(x)