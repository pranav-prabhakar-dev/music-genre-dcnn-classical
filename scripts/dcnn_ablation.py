"""Optional: DCNN architecture study (paper Tables V.1 / V.2), averaged over several seeds."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.dcnn import ConvSpec as S  # noqa: E402
from src.features import load_dataset  # noqa: E402
from src.train import predict, train_dcnn  # noqa: E402

# Each layer: S(filters, kernel, dilation, pool, dropout, padding)
CONFIGS = {
    "1,8,4,32 | 16,4,2,8": [S(1, 8, 4, 32, 0.0), S(16, 4, 2, 8, 0.0, "valid")],
    "8,8,4,32 | 16,4,2,8": [S(8, 8, 4, 32, 0.0), S(16, 4, 2, 8, 0.0, "valid")],
    "8,8,4,d16 | 16,8,2,d4": [S(8, 8, 4, 16), S(16, 8, 2, 4, 0.5, "valid")],
    "8,16,8,d32 | 24,16,2,d4 (final)": [S(8, 16, 8, 32), S(24, 16, 2, 4, 0.5, "valid")],
    "8,64,8,d32 | 24,16,2,d4": [S(8, 64, 8, 32), S(24, 16, 2, 4, 0.5, "valid")],
    "16,64,8,d32 | 64,16,2,d4": [S(16, 64, 8, 32), S(64, 16, 2, 4, 0.5, "valid")],
    "3-layer: 8,16,8,d32 | 24,8,2,d2 | 24,4,2,d2": [S(8, 16, 8, 32), S(24, 8, 2, 2),
                                                    S(24, 4, 2, 2, 0.5, "valid")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    d = load_dataset()
    rows = []
    for name, specs in CONFIGS.items():
        tr, te = [], []
        for seed in range(args.seeds):
            m, sc, _ = train_dcnn(d["X_train"], d["y_train"], d["X_test"], d["y_test"],
                                  specs=specs, epochs=args.epochs, seed=seed, verbose=False)
            tr.append((predict(m, sc, d["X_train"]).argmax(1) == d["y_train"]).mean())
            te.append((predict(m, sc, d["X_test"]).argmax(1) == d["y_test"]).mean())
        rows.append({"config": name, "train_acc": np.mean(tr),
                     "test_acc_mean": np.mean(te), "test_acc_std": np.std(te)})
        print(f"  {name:<44} train {np.mean(tr):.3f}  test {np.mean(te):.3f} +/- {np.std(te):.3f}", flush=True)
    C.ensure_dirs()
    pd.DataFrame(rows).to_csv(C.RESULTS / "dcnn_ablation.csv", index=False)


if __name__ == "__main__":
    main()
