"""Training loop for the Dilated CNN."""
import random

import numpy as np
import torch
import torch.nn as nn

from .dcnn import DilatedCNN, Standardizer, device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate(model, X, y, loss_fn, dev):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(dev))
        loss = loss_fn(logits, torch.from_numpy(y).to(dev)).item()
        pred = logits.argmax(1).cpu().numpy()
    return loss, (pred == y).mean(), pred


def train_dcnn(X_train, y_train, X_test, y_test, specs=None, epochs=150, lr=1e-3,
               batch_size=32, weight_decay=0.0, seed=42, verbose=True):
    """Train with Adam (lr 1e-3) for a fixed number of epochs; the test set is only monitored.

    Returns (model, scaler, history).
    """
    set_seed(seed)
    dev = device()
    scaler = Standardizer().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    ytr, yte = y_train.astype(np.int64), y_test.astype(np.int64)

    model = DilatedCNN(specs).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    hist = {k: [] for k in ("train_loss", "test_loss", "train_acc", "test_acc")}
    g = torch.Generator().manual_seed(seed)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g).numpy()
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            xb, yb = torch.from_numpy(Xtr[b]).to(dev), torch.from_numpy(ytr[b]).to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        tr_loss, tr_acc, _ = _evaluate(model, Xtr, ytr, loss_fn, dev)
        te_loss, te_acc, _ = _evaluate(model, Xte, yte, loss_fn, dev)
        for k, v in zip(hist, (tr_loss, te_loss, tr_acc, te_acc)):
            hist[k].append(float(v))
        if verbose and (ep % 25 == 0 or ep == 1):
            print(f"  epoch {ep:4d}  train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"test loss {te_loss:.3f} acc {te_acc:.3f}")

    model.eval()
    return model.cpu(), scaler, hist


def predict(model, scaler, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(scaler.transform(X)))
        return torch.softmax(logits, 1).numpy()
