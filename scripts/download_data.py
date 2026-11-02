"""Download GTZAN (genres.tar.gz, ~1.2 GB) from the Hugging Face mirror and extract the 5 genres used."""
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"


def main():
    raw = C.DATA_RAW.parent
    raw.mkdir(parents=True, exist_ok=True)
    tar_path = raw / "genres.tar.gz"
    if all(len(list((C.DATA_RAW / g).glob(f"{g}.*.wav"))) == 100 for g in C.GENRES):
        print("GTZAN already extracted at", C.DATA_RAW)
        return
    if not tar_path.exists():
        print(f"Downloading {URL}\n  -> {tar_path}")

        def progress(blocks, bsize, total):
            done = blocks * bsize
            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(URL, tar_path, reporthook=progress)
        print()
    print("Extracting genres:", ", ".join(C.GENRES))
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and not Path(m.name).name.startswith("._")
                   and Path(m.name).parts[:2] in {("genres", g) for g in C.GENRES}]
        tf.extractall(raw, members=members, filter="data")
    for g in C.GENRES:
        print(f"  {g:<10} {len(list((C.DATA_RAW / g).glob(f'{g}.*.wav')))} clips")


if __name__ == "__main__":
    main()
