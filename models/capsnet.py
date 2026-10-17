"""
Spatial-Temporal Representation Learning via GTN, LSTM, and Capsule Routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GTNLayer(nn.Module):
    """
    Graph Transformation Network Layer for spatial feature transformation.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


class SpatialTemporalCapsNet(nn.Module):
    """
    Combines GTN (Spatial), LSTM (Temporal), and Capsule squash non-linearity
    to produce highly discriminative 64-dimensional feature embeddings.
    """
    def __init__(self, input_dim, hidden_dim=32, num_capsules=8, cap_dim=8, num_classes=2):
        super().__init__()
        self.num_caps = num_capsules
        self.cap_dim = cap_dim
        
        self.gtn = GTNLayer(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.caps_proj = nn.Linear(hidden_dim, num_capsules * cap_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_capsules * cap_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def squash(self, s, eps=1e-8):
        """
        Non-linear squashing activation function that maps capsule lengths to [0, 1).
        v_j = (||s_j||^2 / (1 + ||s_j||^2)) * (s_j / ||s_j||)
        """
        norm_squared = torch.sum(s ** 2, dim=-1, keepdim=True)
        scale = norm_squared / (1.0 + norm_squared)
        unit = s / torch.sqrt(norm_squared + eps)
        return scale * unit

    def extract_features(self, x):
        spatial = F.relu(self.gtn(x))
        temporal, _ = self.lstm(spatial.unsqueeze(1))
        caps = self.caps_proj(temporal.squeeze(1)).view(-1, self.num_caps, self.cap_dim)
        v = self.squash(caps)
        return v.view(x.size(0), -1)

    def forward(self, x):
        caps_vec = self.extract_features(x)
        return self.classifier(caps_vec)
