# Music Genre Classification using Dilated CNN and Classical ML

PES University UE24CS352A Machine Learning mini project

## Project Team

| Name | SRN |
|---|---|
| Haneesh E | PES1UG24AM471 |
| Pranav Prabhakar | PES1UG24AM463 |

This project implements and extends the approach described in
[*Combining CNN and Classical Algorithms for Music Genre Classification*](https://cs229.stanford.edu/proj2018/report/19.pdf)
(Li, Xue & Zhang, Stanford CS229, 2018).

## Project Description
Given a 30-second music clip, predict its genre.

The core idea is to use a small neural network as a **feature extractor** for classical machine learning:

1. Turn each clip into **MFCCs**: a 20 × 1290 matrix that summarises the timbre of the sound over time.
2. Train a small **1D dilated CNN** to classify genres from the MFCCs. Dilated convolutions skip frames, so a few
   layers can see long stretches of the clip (about 2.8 s per output) with only 5,973 parameters.
3. Throw away the CNN's final classification layer and keep the **activations of its two convolution layers**.
   These are compact, learned descriptions of the clip (320 and 48 numbers instead of 25,800).
4. Train four **classical classifiers** on those learned features, then compare them against the same
   classifiers trained on raw MFCCs and on a PCA projection of them.

The question being tested: *do features learned by a CNN make classical algorithms better, and can a classical
algorithm on top of CNN features beat the CNN itself?*

## Selected genres
- classical
- hiphop
- metal
- pop
- blues

We use these 5 genres of the GTZAN dataset: 100 clips each, 500 clips in total. A single stratified 80/20 split
(400 train / 100 test, `random_state=42`) is fixed and shared by every model and feature set, so all comparisons are fair.

## Pipeline
1. **Load audio:** 30 s WAV clips from GTZAN at 22,050 Hz.
2. **MFCC features:** 20 coefficients (`n_fft=2048`, `hop=512`), fixed to 1290 frames → a 20 × 1290 matrix per clip.
   Each coefficient is standardised using training-set statistics only.
3. **Train the dilated CNN** (PyTorch, the paper's final configuration):

   | Block | Layers | Output |
   |---|---|---|
   | 1 | Conv1d(20→8, kernel 16, dilation 8) → ReLU → AvgPool(32) → Dropout(0.5) → BatchNorm | 8 × 40 = **320 features** |
   | 2 | Conv1d(8→24, kernel 16, dilation 2) → ReLU → AvgPool(4) → Dropout(0.5) → BatchNorm | 24 × 2 = **48 features** |
   | Head | Linear(48→5) + softmax | genre probabilities |

   Adam, learning rate 1e-3, batch 32, 150 epochs. Training length is fixed; the test set is never used to pick a model.
4. **Extract features** from both convolution blocks for every clip.
5. **Train four classical classifiers** (scikit-learn, each as `StandardScaler → [PCA] → classifier`):
   - **LR:** multinomial (softmax) logistic regression
   - **GDA:** Gaussian discriminant analysis with shared covariance (`LinearDiscriminantAnalysis`)
   - **RF:** random forest, 300 trees, `max_depth=7`
   - **SVM:** RBF kernel, `gamma='scale'`, one-vs-one

   on four inputs:
   - flattened raw MFCCs (25,800 values)
   - PCA of the raw MFCCs (50 components)
   - CNN block 1 activations (320)
   - CNN block 2 activations (48)
6. **Evaluate** test accuracy and confusion matrices, then check robustness over several training seeds.

## Results

### Main comparison (test accuracy, seed 42)

| Input features (dim) | LR | GDA | RF | SVM |
|---|---|---|---|---|
| Raw MFCC (25,800) | 0.84 | 0.83 | 0.74 | 0.79 |
| PCA of raw MFCC (50) | 0.85 | 0.79 | 0.82 | 0.83 |
| CNN block 1 (320) | 0.88 | 0.61 | 0.87 | 0.88 |
| CNN block 2 (48) | **0.89** | 0.86 | 0.87 | 0.86 |

The dilated CNN on its own scores **0.89** on test (0.99 on train).

### Averaged over 5 CNN training seeds

A 100-clip test set is noisy (one clip = 1 point), so we retrained the CNN with 5 different seeds and report mean ± std.

| Features | CNN alone | LR | GDA | RF | SVM |
|---|---|---|---|---|---|
| CNN block 1 | 0.872 ± 0.028 | 0.856 ± 0.018 | 0.596 ± 0.080 | 0.860 ± 0.024 | **0.894 ± 0.024** |
| CNN block 2 | | 0.866 ± 0.034 | 0.846 ± 0.015 | 0.850 ± 0.045 | 0.872 ± 0.018 |

### Key findings
1. **CNN features help the classical algorithms.** Every classifier except GDA does better on CNN features than on
   raw or PCA MFCCs. The biggest gains are RF (0.74 → 0.87) and SVM (0.79 → 0.88).
2. **An SVM on CNN features beats the CNN itself.** Averaged over seeds, the RBF-SVM on block 1 features reaches
   0.894, above the CNN it was extracted from (0.872). This confirms the paper's main claim.
3. **The features are far smaller.** 48 learned numbers match or beat 25,800 raw values.
4. **GDA fails on block 1 features** (0.61, as in the paper). Its single shared Gaussian is a poor fit for 320
   correlated, non-negative activations estimated from only 400 clips.
5. **The paper's very poor SVM results on raw MFCCs (0.28 / 0.21) come from preprocessing, not from SVMs.**
   We reproduce them (train 1.00, test 0.39 / 0.22) by feeding *unscaled* MFCCs with `gamma='auto'`, the old
   scikit-learn default. The RBF kernel becomes far too narrow and the SVM just memorises the training set.
   With standardised inputs and `gamma='scale'` the same SVM reaches 0.79–0.83.

   | SVM input | Standardised | gamma | Train | Test |
   |---|---|---|---|---|
   | Raw MFCC | no | auto | 1.00 | 0.39 |
   | PCA of raw | no | auto | 1.00 | 0.22 |
   | Raw MFCC | no | scale | 0.85 | 0.76 |
   | PCA of raw | no | scale | 0.87 | 0.76 |
   | Raw MFCC | yes | scale | 0.99 | 0.79 |
   | PCA of raw | yes | scale | 0.93 | 0.83 |

### CNN architecture study
We re-ran the paper's architecture experiments, averaging 3 seeds each. Layer notation is
filters, kernel, dilation, pool; `d` means dropout 0.5.

| Configuration | Train | Test |
|---|---|---|
| 1,8,4,32 \| 16,4,2,8 | 0.91 | 0.64 ± 0.04 |
| 8,8,4,32 \| 16,4,2,8 | 1.00 | **0.91 ± 0.01** |
| 8,8,4,d16 \| 16,8,2,d4 | 0.99 | 0.88 ± 0.02 |
| 8,16,8,d32 \| 24,16,2,d4 (paper's final, used above) | 0.99 | 0.87 ± 0.03 |
| 8,64,8,d32 \| 24,16,2,d4 | 0.99 | 0.85 ± 0.01 |
| 16,64,8,d32 \| 64,16,2,d4 | 1.00 | 0.86 ± 0.01 |
| 3 blocks: 8,16,8,d32 \| 24,8,2,d2 \| 24,4,2,d2 | 0.98 | 0.87 ± 0.01 |

A single first-layer filter is far too few, which agrees with the paper. Every other configuration lands between 0.85
and 0.91. Unlike the paper, we found that larger filters did not help and a 3-block network did not overfit. We kept
the paper's final architecture for the main experiments so the comparison stays faithful.

Figures (MFCC heatmaps, training curves, confusion matrices, 3D PCA of each feature space, result charts) are in
`results/figures/`. The two-page write-up is [report/writeup.pdf](report/writeup.pdf).

## Dataset layout
`scripts/download_data.py` downloads GTZAN (~1.2 GB, Hugging Face mirror `marsyas/gtzan`) and extracts only the
five selected genres to:

```text
data/
  raw/
    genres/
      blues/       blues.00000.wav ... blues.00099.wav
      classical/
      hiphop/
      metal/
      pop/
  processed/
    mfcc_dataset.npz   (cached MFCCs + the fixed train/test split)
```

The dataset is not committed to the repository.

## Setup (Windows PowerShell)
Requires Python 3.10+ (tested on 3.13, Windows 11, CPU only).

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS, activate with `source .venv/bin/activate` instead.

## Run
Everything at once (about 5 minutes on a laptop CPU, after the download):
```powershell
python run_all.py            # download, MFCCs, train CNN, classical models
python run_all.py --extras   # also the 5-seed study and architecture study (~20 min more)
```

Or step by step:
```powershell
python scripts/download_data.py    # 0. get GTZAN
python scripts/prepare_data.py     # 1. extract MFCCs and fix the split
python scripts/train_dcnn.py       # 2. train the dilated CNN
python scripts/run_classical.py    # 3. classical models on all four feature sets + SVM check
python scripts/seed_study.py       # optional: 5-seed robustness study
python scripts/dcnn_ablation.py    # optional: architecture study
```

Outputs are saved under `models/` (trained CNN and classical models) and `results/` (CSV/Markdown tables, logs, figures).

## Live demo
```powershell
python app.py                # then open http://127.0.0.1:7860
```

Pick a held-out test clip, or upload or record your own audio. The app shows the clip's MFCCs, the CNN's genre
probabilities, and the predictions of all eight classical models (4 algorithms × 2 CNN blocks).
It needs the data from steps 0–1 and the models in `models/` (already included).

## Repository layout
```text
src/
  config.py        paths, genres, MFCC settings, seed
  features.py      WAV -> 20x1290 MFCC; fixed stratified 80/20 split
  dcnn.py          dilated CNN, feature extraction, checkpoint I/O
  train.py         CNN training loop
  classical.py     LR / GDA / RF / SVM pipelines
  plots.py         all figures
scripts/           the runnable steps listed under Run
app.py             Gradio live demo
run_all.py         runs the whole pipeline
report/            write-up source and PDF
models/            trained models used by the demo
results/           tables, logs and figures
```

## Differences from the paper
- PyTorch instead of Keras, and ReLU activations (the paper does not name its activation).
- Inputs to the classical models are standardised, which fixes the paper's SVM failure (finding 5).
- Our random 400/100 split differs from the paper's, so individual numbers differ by a few points. That is why we
  also report results averaged over several seeds.
- Our architecture study disagrees with the paper on whether larger filters and a third block help.

## Notes
- All numbers above were produced by the scripts in this repository; none are copied from the paper.
- The same train/test indices are used for every feature set and classifier.
