"""Run the full pipeline end to end: download -> MFCC -> DCNN -> classical comparison.

Add --extras to also run the multi-seed study and the DCNN architecture ablation (~20 min on CPU).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["download_data.py", "prepare_data.py", "train_dcnn.py", "run_classical.py"]
EXTRAS = ["seed_study.py", "dcnn_ablation.py"]

if __name__ == "__main__":
    for script in STEPS + (EXTRAS if "--extras" in sys.argv else []):
        print(f"\n=== {script} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)
