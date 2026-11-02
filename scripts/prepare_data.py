"""Step 1: extract (20 x 1290) MFCC arrays for the 500 clips and fix the 80/20 train/test split."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, plots  # noqa: E402
from src.features import build_dataset  # noqa: E402


def main():
    t = time.time()
    print("Extracting MFCC features ...")
    d = build_dataset()
    print(f"Train: {d['X_train'].shape}  Test: {d['X_test'].shape}  "
          f"({time.time() - t:.0f}s) -> {C.MFCC_CACHE.relative_to(C.ROOT)}")
    print("Saved", plots.mfcc_heatmaps(d["X_train"], d["y_train"]).relative_to(C.ROOT))


if __name__ == "__main__":
    main()
