"""1D Dilated CNN over MFCC frames, usable both as a classifier and as a feature extractor."""
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from . import config as C


@dataclass
class ConvSpec:
    filters: int
    kernel: int
    dilation: int
    pool: int
    dropout: float = 0.5
    padding: str = "same"   # "same" or "valid"


# Final architecture from the paper (Table V.2, row "8,16,8,d32 | 24,16,2,d4").
# Layer-1 activations: 8 filters x (1290/32 = 40) steps = 320 features.
# Layer-2 activations: 24 filters x ((40 - 15*2)/4 = 2) steps = 48 features.
DEFAULT_SPECS = [
    ConvSpec(filters=8, kernel=16, dilation=8, pool=32, padding="same"),
    ConvSpec(filters=24, kernel=16, dilation=2, pool=4, padding="valid"),
]


class DilatedUnit(nn.Module):
    """Dilated Conv1d -> ReLU -> AvgPool -> Dropout -> BatchNorm (one 'dilated convolution unit')."""

    def __init__(self, in_ch: int, spec: ConvSpec):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, spec.filters, spec.kernel,
                              dilation=spec.dilation, padding=spec.padding)
        self.act = nn.ReLU()
        self.pool = nn.AvgPool1d(spec.pool)
        self.drop = nn.Dropout(spec.dropout)
        self.bn = nn.BatchNorm1d(spec.filters)

    def forward(self, x):
        return self.bn(self.drop(self.pool(self.act(self.conv(x)))))


class DilatedCNN(nn.Module):
    def __init__(self, specs=None, n_mfcc: int = C.N_MFCC, n_frames: int = C.N_FRAMES,
                 n_classes: int = len(C.GENRES)):
        super().__init__()
        self.specs = list(specs or DEFAULT_SPECS)
        units, in_ch = [], n_mfcc
        for s in self.specs:
            units.append(DilatedUnit(in_ch, s))
            in_ch = s.filters
        self.units = nn.ModuleList(units)
        with torch.no_grad():
            flat = self._forward_units(torch.zeros(1, n_mfcc, n_frames))[-1].flatten(1).shape[1]
        self.fc = nn.Linear(flat, n_classes)   # softmax is folded into the cross-entropy loss

    def _forward_units(self, x):
        outs = []
        for u in self.units:
            x = u(x)
            outs.append(x)
        return outs

    def forward(self, x):
        return self.fc(self._forward_units(x)[-1].flatten(1))

    @torch.no_grad()
    def layer_features(self, x):
        """Return the flattened activations of every dilated unit (list, one per layer)."""
        return [o.flatten(1) for o in self._forward_units(x)]


class Standardizer:
    """Per-MFCC-coefficient standardisation fitted on the training set."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, X):
        self.mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std = (X.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: DilatedCNN, scaler: Standardizer, path=C.DCNN_CKPT, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "specs": [asdict(s) for s in model.specs],
                "mean": scaler.mean, "std": scaler.std,
                "extra": extra or {}}, path)


def load_checkpoint(path=C.DCNN_CKPT):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DilatedCNN([ConvSpec(**s) for s in ckpt["specs"]])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, Standardizer(ckpt["mean"], ckpt["std"]), ckpt.get("extra", {})


def extract_features(model: DilatedCNN, scaler: Standardizer, X: np.ndarray, batch: int = 64):
    """Run MFCC arrays through the trained DCNN and return [layer1_feats, layer2_feats, ...]."""
    model.eval()
    dev = next(model.parameters()).device
    Xs = scaler.transform(X)
    per_layer = None
    for i in range(0, len(Xs), batch):
        feats = model.layer_features(torch.from_numpy(Xs[i:i + batch]).to(dev))
        feats = [f.cpu().numpy() for f in feats]
        per_layer = feats if per_layer is None else [np.vstack([a, b]) for a, b in zip(per_layer, feats)]
    return per_layer
