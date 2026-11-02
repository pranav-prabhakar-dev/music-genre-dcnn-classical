# Combining CNN and Classical Algorithms for Music Genre Classification

UE24CS352A Machine Learning mini-project. This is a re-implementation and extension of
[Li, Xue & Zhang, *Combining CNN and Classical Algorithms for Music Genre Classification* (CS229, 2018)](https://cs229.stanford.edu/proj2018/report/19.pdf).

**Idea:** train a small **1D Dilated CNN (DCNN)** on MFCC features of music clips, then use the
activations of its convolution layers as **learned features** for four classical classifiers:
Softmax Logistic Regression (LR), Gaussian Discriminant Analysis (GDA), Random Forest (RF), and RBF-SVM.
We compare those classifiers on four inputs: the raw MFCCs, a PCA projection of them, DCNN layer 1 activations, and DCNN layer 2 activations.

## Results (5 GTZAN genres, fixed 400/100 stratified split)

Test accuracy (seed 42, `scripts/run_classical.py`):

| Input features (dim)  | LR   | GDA  | RF   | SVM  |
|-----------------------|------|------|------|------|
| Raw MFCC (25800)      | 0.84 | 0.83 | 0.74 | 0.79 |
| PCA of raw MFCC (50)  | 0.85 | 0.79 | 0.82 | 0.83 |
| DCNN layer 1 (320)    | 0.88 | 0.61 | 0.87 | 0.88 |
| DCNN layer 2 (48)     | **0.89** | 0.86 | 0.87 | 0.86 |
| *DCNN alone*          | *0.89* |    |      |      |

Mean ± std over 5 DCNN training seeds (`scripts/seed_study.py`):

| | DCNN alone | LR | GDA | RF | SVM |
|---|---|---|---|---|---|
| Layer 1 features | 0.872 ± 0.028 | 0.856 ± 0.018 | 0.596 ± 0.080 | 0.860 ± 0.024 | **0.894 ± 0.024** |
| Layer 2 features | | 0.866 ± 0.034 | 0.846 ± 0.015 | 0.850 ± 0.045 | 0.872 ± 0.018 |

DCNN architecture study, mean test accuracy over 3 seeds (`scripts/dcnn_ablation.py`; the layer
notation is filters, kernel, dilation, pool, and `d` means dropout 0.5):

| Configuration | Train | Test |
|---|---|---|
| 1,8,4,32 \| 16,4,2,8 | 0.91 | 0.64 ± 0.04 |
| 8,8,4,32 \| 16,4,2,8 | 1.00 | **0.91 ± 0.01** |
| 8,8,4,d16 \| 16,8,2,d4 | 0.99 | 0.88 ± 0.02 |
| 8,16,8,d32 \| 24,16,2,d4 (paper's final, used here) | 0.99 | 0.87 ± 0.03 |
| 8,64,8,d32 \| 24,16,2,d4 | 0.99 | 0.85 ± 0.01 |
| 16,64,8,d32 \| 64,16,2,d4 | 1.00 | 0.86 ± 0.01 |
| 3 units: 8,16,8,d32 \| 24,8,2,d2 \| 24,4,2,d2 | 0.98 | 0.87 ± 0.01 |

Key findings:
* Every classical algorithm except GDA-on-layer-1 does better on DCNN features than on raw or PCA MFCCs
  (RF: 0.74 → 0.87, SVM: 0.79 → 0.88).
* **RBF-SVM on DCNN layer 1 beats the DCNN it was extracted from** when averaged over seeds (0.894 vs 0.872),
  which supports the paper's claim.
* GDA collapses on layer 1 features (0.61 here and in the paper). Its shared-covariance Gaussian
  model is a poor fit for 320 correlated, non-negative pooled activations from only 400 samples.
* The paper's very poor SVM numbers on raw/PCA input (0.28 / 0.21) are **not** inherent to SVMs.
  We reproduce them exactly (train 1.00, test 0.39 / 0.22) by using unscaled MFCCs with
  `gamma='auto'`, scikit-learn's default before 2019. The RBF kernel is then far too narrow and the
  SVM memorises the training set. With standardised inputs and `gamma='scale'` the SVM reaches 0.79–0.83.
* In the architecture study, a single first-layer filter is far too few (0.64). All other configurations land
  at 0.85–0.91. Unlike the paper, larger filters did not help and a 3-unit network did not overfit.
  We kept the paper's final architecture for comparability.

Figures are written to `results/figures/` (MFCC heatmaps, training curves, confusion matrices,
3D PCA of each feature space, and result charts).

## Repository layout

```
src/
  config.py        paths, genres, MFCC settings, seed
  features.py      wav -> 20x1290 MFCC; fixed stratified 80/20 split (cached .npz)
  dcnn.py          PyTorch Dilated CNN (+ feature extraction, checkpoint I/O)
  train.py         training loop (Adam, lr 1e-3, 150 epochs, batch 32)
  classical.py     LR / GDA / RF / SVM sklearn pipelines
  plots.py         all figures
scripts/
  download_data.py  downloads GTZAN (~1.2 GB) and extracts the 5 genres
  prepare_data.py   step 1 - MFCC extraction
  train_dcnn.py     step 2 - train + save the DCNN
  run_classical.py  step 3 - Table V.3 reproduction + SVM gamma check + figures
  seed_study.py     optional - 5-seed robustness study
  dcnn_ablation.py  optional - architecture study (paper Tables V.1/V.2)
app.py             Gradio live demo
run_all.py         runs steps 0-3 (add --extras for the optional studies)
report/            write-up source + PDF
models/            trained DCNN + classical models used by the demo
results/           CSV/Markdown tables, logs, figures
```

## Setup

Requires Python 3.10+ (tested on 3.13, Windows 11, CPU only; the whole pipeline runs in about 5 minutes).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run

```bash
python run_all.py               # download data, extract MFCCs, train DCNN, run classical models
python run_all.py --extras      # + seed study and DCNN ablation (~20 min more)
```

Or step by step:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_dcnn.py --epochs 150
python scripts/run_classical.py
```

## Live demo

```bash
python app.py                   # open http://127.0.0.1:7860
```

Pick any held-out test clip, or upload or record your own audio. The app shows the MFCC input, the DCNN's
genre probabilities, and the predictions of all eight classical models (4 algorithms × 2 DCNN layers).
The demo needs `data/` (run steps 0–1 first) and the models in `models/`.

## Method details

* **Data:** GTZAN, genres classical, hiphop, metal, pop, blues; 100 × 30 s clips each (500 total).
  Stratified 80/20 split (`random_state=42`), fixed for every experiment.
* **Features:** librosa MFCC, 20 coefficients, `n_fft=2048`, `hop=512`, 22.05 kHz, cropped/tiled to 1290
  frames → a 20 × 1290 matrix (25800 values when flattened).
* **DCNN:** the paper's final configuration (`8,16,8,d32 | 24,16,2,d4`):
  `Conv1d(20→8, k=16, dilation=8, same) → ReLU → AvgPool(32) → Dropout(0.5) → BatchNorm` (layer 1, 8×40 = 320 features),
  `Conv1d(8→24, k=16, dilation=2, valid) → ReLU → AvgPool(4) → Dropout(0.5) → BatchNorm` (layer 2, 24×2 = 48 features),
  then `Linear(48→5)` + softmax. 5,973 parameters. Inputs are standardised per MFCC coefficient using
  training-set statistics. The model is trained for a fixed number of epochs; the test set is never used for
  model selection.
* **Classical models:** each is a `StandardScaler → [PCA(50)] → classifier` pipeline.
  LR = multinomial logistic regression; GDA = shared-covariance Gaussian discriminant (LDA);
  RF = 300 Gini trees with `max_depth=7`; SVM = RBF kernel, `gamma='scale'`, one-vs-one.

## Differences from the paper

* PyTorch instead of Keras, and ReLU activations (the paper does not name its activation).
* Features are standardised before the classical models. This fixes the paper's SVM failure (see above).
* The random 400/100 split differs, so per-cell numbers differ by a few points. Results are reported over
  several seeds as well, because one 100-clip test set moves ±3 points between runs.
