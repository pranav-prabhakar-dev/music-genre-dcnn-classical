"""Audio -> 2D MFCC feature extraction and the fixed train/test split."""
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

from . import config as C


def fix_length(mfcc: np.ndarray, n_frames: int = C.N_FRAMES) -> np.ndarray:
    """Crop or tile-pad an (n_mfcc, T) array along time to exactly n_frames."""
    if mfcc.shape[1] >= n_frames:
        return mfcc[:, :n_frames]
    reps = int(np.ceil(n_frames / mfcc.shape[1]))
    return np.tile(mfcc, (1, reps))[:, :n_frames]


def mfcc_from_signal(y: np.ndarray, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=C.N_MFCC,
                                n_fft=C.N_FFT, hop_length=C.HOP_LENGTH)
    return fix_length(mfcc).astype(np.float32)


def mfcc_from_file(path, offset: float = 0.0) -> np.ndarray:
    """Load (up to) 30 s of audio starting at `offset` seconds and return a (20, 1290) MFCC."""
    y, sr = librosa.load(path, sr=C.SAMPLE_RATE, mono=True,
                         offset=offset, duration=C.DURATION)
    return mfcc_from_signal(y, sr)


def build_dataset(verbose: bool = True):
    """Extract MFCCs for every clip of the 5 genres and cache them with a fixed 80/20 split."""
    X, y, files = [], [], []
    for label, genre in enumerate(C.GENRES):
        # skip macOS "._*" metadata files that ship inside the GTZAN tarball
        paths = sorted(p for p in (C.DATA_RAW / genre).glob(f"{genre}.*")
                       if p.suffix in (".wav", ".au"))
        if not paths:
            raise FileNotFoundError(f"No audio found in {C.DATA_RAW / genre}. "
                                    "Run `python scripts/download_data.py` first.")
        for p in paths:
            X.append(mfcc_from_file(p))
            y.append(label)
            files.append(f"{genre}/{p.name}")
        if verbose:
            print(f"  {genre:<10} {len(paths)} clips")

    X, y, files = np.stack(X), np.array(y), np.array(files)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=C.TEST_SIZE,
                                           stratify=y, random_state=C.SEED)
    C.ensure_dirs()
    np.savez_compressed(C.MFCC_CACHE, X=X, y=y, files=files,
                        train_idx=np.sort(train_idx), test_idx=np.sort(test_idx))
    return load_dataset()


def load_dataset():
    """Return X_train, X_test, y_train, y_test (MFCC arrays of shape (n, 20, 1290)) plus file names."""
    if not C.MFCC_CACHE.exists():
        raise FileNotFoundError(f"{C.MFCC_CACHE} missing. Run `python scripts/prepare_data.py` first.")
    d = np.load(C.MFCC_CACHE, allow_pickle=False)
    tr, te = d["train_idx"], d["test_idx"]
    return {
        "X_train": d["X"][tr], "X_test": d["X"][te],
        "y_train": d["y"][tr], "y_test": d["y"][te],
        "files_train": d["files"][tr], "files_test": d["files"][te],
    }
