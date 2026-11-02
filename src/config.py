"""Project-wide constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "genres"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# The paper uses 5 of the 10 GTZAN genres.
GENRES = ["classical", "hiphop", "metal", "pop", "blues"]

# Audio / MFCC settings: 30 s at 22.05 kHz with hop 512 gives ~1293 frames;
# we fix every clip to 1290 frames so that 20 x 1290 = 25800 features (as in the paper).
SAMPLE_RATE = 22050
DURATION = 30.0
N_MFCC = 20
HOP_LENGTH = 512
N_FFT = 2048
N_FRAMES = 1290

TEST_SIZE = 0.2
SEED = 42

MFCC_CACHE = DATA_PROCESSED / "mfcc_dataset.npz"
DCNN_CKPT = MODELS / "dcnn.pt"


def ensure_dirs():
    for d in (DATA_PROCESSED, MODELS, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
