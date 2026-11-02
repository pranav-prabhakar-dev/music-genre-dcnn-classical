"""Step 3: train LR / GDA / RF / SVM on four feature sets and reproduce Table V.3 of the paper.

Feature sets: (1) raw flattened MFCC (25800), (2) PCA-50 of raw MFCC,
(3) DCNN layer-1 activations (320), (4) DCNN layer-2 activations (48).
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, plots  # noqa: E402
from src.classical import ALGORITHMS, make_pipeline_for  # noqa: E402
from src.dcnn import extract_features, load_checkpoint  # noqa: E402
from src.features import load_dataset  # noqa: E402

warnings.filterwarnings("ignore", message="Variables are collinear")


def main():
    C.ensure_dirs()
    d = load_dataset()
    ytr, yte = d["y_train"], d["y_test"]
    model, scaler, extra = load_checkpoint()
    l1_tr, l2_tr = extract_features(model, scaler, d["X_train"])
    l1_te, l2_te = extract_features(model, scaler, d["X_test"])
    raw_tr = d["X_train"].reshape(len(ytr), -1)
    raw_te = d["X_test"].reshape(len(yte), -1)

    # name -> (train features, test features, PCA components or None)
    feature_sets = {
        f"Raw MFCC ({raw_tr.shape[1]})": (raw_tr, raw_te, None),
        "PCA of raw (50)": (raw_tr, raw_te, 50),
        f"DCNN layer 1 ({l1_tr.shape[1]})": (l1_tr, l1_te, None),
        f"DCNN layer 2 ({l2_tr.shape[1]})": (l2_tr, l2_te, None),
    }
    rows, test_acc = [], {}
    for feat, (Ftr, Fte, pca) in feature_sets.items():
        for alg in ALGORITHMS:
            t = time.time()
            pipe = make_pipeline_for(alg, pca, probability=feat.startswith("DCNN")).fit(Ftr, ytr)
            tr = (pipe.predict(Ftr) == ytr).mean()
            pred = pipe.predict(Fte)
            te = (pred == yte).mean()
            rows.append({"features": feat, "algorithm": alg, "train_acc": tr, "test_acc": te})
            test_acc[(feat, alg)] = te
            print(f"  {feat:<22} {alg:<4} train {tr:.3f}  test {te:.3f}  ({time.time() - t:.1f}s)")
            if feat.startswith("DCNN"):
                tag = "l1" if "layer 1" in feat else "l2"
                joblib.dump(pipe, C.MODELS / f"{alg.lower()}_{tag}.joblib")
                plots.confusion(yte, pred, f"{alg} on {feat} ({te:.2f})",
                                f"confusion_{alg.lower()}_{tag}.png")

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / "classical_results.csv", index=False)
    table = res.pivot(index="features", columns="algorithm", values="test_acc")
    table = table.loc[list(feature_sets), ALGORITHMS]
    wide = res.pivot(index="features", columns="algorithm", values=["train_acc", "test_acc"])
    print("\nTest accuracy (rows = input features):\n", table.round(3).to_string())
    print(f"\nDCNN alone: test acc {extra['test_acc']:.3f}")

    # The paper reports SVM = 0.28 / 0.21 test (1.0 train) on raw / PCA inputs. That pattern is
    # reproduced by the pre-2019 sklearn default gamma='auto' (= 1/n_features, ignoring the feature
    # variance) on unscaled MFCCs: the kernel becomes too narrow and the SVM just memorises.
    svm_check = []
    for scale, gamma in ((False, "scale"), (False, "auto")):
        for feat, pca in (("Raw MFCC (25800)", None), ("PCA of raw (50)", 50)):
            pipe = make_pipeline_for("SVM", pca, scale=scale, svm_gamma=gamma).fit(raw_tr, ytr)
            tr, te = (pipe.predict(raw_tr) == ytr).mean(), (pipe.predict(raw_te) == yte).mean()
            svm_check.append({"features": feat, "standardised": scale, "gamma": gamma,
                              "train_acc": tr, "test_acc": te})
            print(f"  SVM unscaled, gamma={gamma:<5} on {feat:<18} train {tr:.3f}  test {te:.3f}")
    pd.DataFrame(svm_check).to_csv(C.RESULTS / "svm_gamma_check.csv", index=False)
    (C.RESULTS / "classical_results_table.md").write_text(
        "| Features | split | " + " | ".join(ALGORITHMS) + " |\n|---|---|" + "---|" * len(ALGORITHMS) + "\n" +
        "".join(f"| {f} | {s} | " + " | ".join(f"{wide.loc[f, (s + '_acc', a)]:.3f}" for a in ALGORITHMS) + " |\n"
                for f in feature_sets for s in ("train", "test")) +
        f"\nDCNN alone: train {extra['train_acc']:.3f}, test {extra['test_acc']:.3f}\n")

    plots.results_heatmap(table)
    plots.results_bars(table, extra["test_acc"])
    plots.pca_3d({"Raw flattened MFCC": raw_te, "DCNN layer 1": l1_te, "DCNN layer 2": l2_te}, yte)
    lr_l1 = joblib.load(C.MODELS / "lr_l1.joblib")
    plots.pca_3d({"LR on layer 1 - true genre": l1_te}, yte, name="pca_l1_true.png")
    plots.pca_3d({"LR on layer 1 - predicted genre": l1_te}, lr_l1.predict(l1_te),
                 name="pca_l1_pred.png")
    print("Results and figures written to", C.RESULTS.relative_to(C.ROOT))


if __name__ == "__main__":
    main()
