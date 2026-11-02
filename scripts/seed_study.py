"""Optional: repeat DCNN training + classical-on-DCNN-features over several seeds.

With only 100 test clips one run can move by +/-3 points, so this reports mean +/- std.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.classical import ALGORITHMS, make_pipeline_for  # noqa: E402
from src.dcnn import extract_features  # noqa: E402
from src.features import load_dataset  # noqa: E402
from src.train import predict, train_dcnn  # noqa: E402

warnings.filterwarnings("ignore", message="Variables are collinear")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    d = load_dataset()
    ytr, yte = d["y_train"], d["y_test"]
    rows = []
    for seed in range(args.seeds):
        model, scaler, _ = train_dcnn(d["X_train"], ytr, d["X_test"], yte,
                                      epochs=args.epochs, seed=seed, verbose=False)
        dcnn_acc = (predict(model, scaler, d["X_test"]).argmax(1) == yte).mean()
        rows.append({"seed": seed, "features": "DCNN alone", "algorithm": "DCNN", "test_acc": dcnn_acc})
        feats_tr = extract_features(model, scaler, d["X_train"])
        feats_te = extract_features(model, scaler, d["X_test"])
        msg = [f"DCNN {dcnn_acc:.2f}"]
        for li, (Ftr, Fte) in enumerate(zip(feats_tr, feats_te), start=1):
            for alg in ALGORITHMS:
                acc = (make_pipeline_for(alg).fit(Ftr, ytr).predict(Fte) == yte).mean()
                rows.append({"seed": seed, "features": f"DCNN layer {li}", "algorithm": alg, "test_acc": acc})
                msg.append(f"L{li}-{alg} {acc:.2f}")
        print(f"  seed {seed}: " + "  ".join(msg))

    df = pd.DataFrame(rows)
    C.ensure_dirs()
    df.to_csv(C.RESULTS / "seed_study_runs.csv", index=False)
    summary = df.groupby(["features", "algorithm"])["test_acc"].agg(["mean", "std"]).reset_index()
    summary.to_csv(C.RESULTS / "seed_study_summary.csv", index=False)
    print("\nMean +/- std test accuracy over", args.seeds, "seeds:")
    for _, r in summary.iterrows():
        print(f"  {r.features:<14} {r.algorithm:<5} {r['mean']:.3f} +/- {r['std']:.3f}")


if __name__ == "__main__":
    main()
