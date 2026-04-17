import torch
import torch.nn as nn


class CommonEEGClassifier(nn.Module):
    """Shared encoder-classifier wrapper for continuous EEG backbones."""

    def __init__(
        self,
        encoder,
        feature_dim,
        num_classes,
        projection_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        if projection_dim is None:
            raise ValueError("CommonEEGClassifier requires projection_dim for the shared classifier head.")

        self.feature_adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classification_head = nn.Linear(projection_dim, num_classes)

    def encode_features(self, signal, pos=None):
        # Encoder wrappers are responsible for any positional-input handling
        # and pooling so the classifier always sees a pooled [B, E] feature map
        encoded = self.encoder(signal, pos=pos)
        if encoded.dim() != 2:
            raise ValueError(f"Encoder must return pooled [B, E] features, got {tuple(encoded.shape)}")
        return self.feature_adapter(encoded)

    def forward(self, signal, pos=None):
        features = self.encode_features(signal, pos=pos)
        logits = self.classification_head(features)
        return {
            "features": features,
            "logit": logits,
        }

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
        self.encoder.eval()
