"""Step 2: train the 2-layer Dilated CNN on MFCCs and save it (it is later used as a feature extractor)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, plots  # noqa: E402
from src.dcnn import save_checkpoint  # noqa: E402
from src.features import load_dataset  # noqa: E402
from src.train import predict, train_dcnn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    C.ensure_dirs()
    d = load_dataset()
    print(f"Training Dilated CNN for {args.epochs} epochs (Adam, lr={args.lr}) ...")
    model, scaler, hist = train_dcnn(d["X_train"], d["y_train"], d["X_test"], d["y_test"],
                                     epochs=args.epochs, lr=args.lr,
                                     batch_size=args.batch_size, seed=args.seed)
    tr_acc = (predict(model, scaler, d["X_train"]).argmax(1) == d["y_train"]).mean()
    te_pred = predict(model, scaler, d["X_test"]).argmax(1)
    te_acc = (te_pred == d["y_test"]).mean()
    print(f"\nFinal DCNN  train acc {tr_acc:.3f}  test acc {te_acc:.3f}")

    n_params = sum(p.numel() for p in model.parameters())
    save_checkpoint(model, scaler, extra={"train_acc": float(tr_acc), "test_acc": float(te_acc),
                                          "epochs": args.epochs, "seed": args.seed,
                                          "n_params": n_params})
    (C.RESULTS / "dcnn_history.json").write_text(json.dumps(hist))
    (C.RESULTS / "dcnn_metrics.json").write_text(json.dumps(
        {"train_acc": float(tr_acc), "test_acc": float(te_acc), "n_params": n_params}, indent=2))
    plots.training_curves(hist)
    plots.confusion(d["y_test"], te_pred, f"DCNN (test acc {te_acc:.2f})", "confusion_dcnn.png")
    print("Saved model to", C.DCNN_CKPT.relative_to(C.ROOT), f"({n_params} parameters)")


if __name__ == "__main__":
    main()
