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
Given a 30-second music clip, the goal is to predict its genre.

The idea is to combine a neural network with classical machine learning:

1. Convert each song into **MFCCs**, a standard numeric summary of how the sound changes over time.
2. Train a small **CNN** (convolutional neural network) to recognise the genre from the MFCCs.
3. Use the CNN as a **feature extractor**: take the values from its inner layers, which are a short learned
   summary of the song.
4. Feed those features into classical classifiers (**Logistic Regression, GDA, Random Forest, SVM**) and check
   whether they do better than with the plain MFCCs.

The question: *do features learned by a CNN help classical algorithms, and can they even beat the CNN itself?*

## Selected genres
- classical
- hiphop
- metal
- pop
- blues

We use 500 tracks from the GTZAN dataset (100 per genre) and an 80/20 train/test split
(400 songs to train, 100 to test). The same split is used for every model so the comparison is fair.

## Pipeline
1. Load the 30-second WAV files.
2. Extract 20 MFCC coefficients per time step and normalise them.
3. Train a 1D dilated CNN on the MFCCs.
4. Take the outputs of the CNN's two hidden layers as features.
5. Train Logistic Regression, GDA, Random Forest and SVM on four kinds of input:
   - raw MFCCs,
   - MFCCs reduced with PCA,
   - CNN layer 1 features,
   - CNN layer 2 features.
6. Compare test accuracy across all models.

## Results
Test accuracy on the 100 held-out songs:

| Input features | Logistic Regression | GDA | Random Forest | SVM |
|---|---|---|---|---|
| Raw MFCCs | 84% | 83% | 74% | 79% |
| MFCCs + PCA | 85% | 79% | 82% | 83% |
| CNN layer 1 | 88% | 61% | 87% | 88% |
| CNN layer 2 | **89%** | 86% | 87% | 86% |

The CNN on its own scores **89%**.

### Key findings
- **CNN features make classical models better.** Random Forest goes from 74% to 87% and SVM from 79% to 88%.
- **SVM on CNN features can beat the CNN.** Averaged over 5 training runs, SVM on layer 1 scores 89.4%
  vs 87.2% for the CNN alone, which supports the paper's main claim.
- **Small but powerful features.** The CNN reduces each song from 25,800 numbers to just 48, and accuracy goes up.
- **GDA does not work well on layer 1 features** (61%), the same as in the paper.
- **We explain the paper's poor SVM scores.** The paper reported only 28% / 21% for SVM on raw MFCCs. We got the
  same failure by skipping feature scaling and using an old SVM default setting; set up properly, SVM reaches 79–83%.
- **Changing the CNN's size matters little.** We also tested 7 CNN designs. All but the smallest scored 85–91%.

Charts and confusion matrices are in `results/figures/`. More detail is in the
[two-page write-up](report/writeup.pdf).

## Dataset layout
The download script fetches GTZAN (~1.2 GB) and keeps only the five genres:

```text
data/
  raw/
    genres/
      blues/
      classical/
      hiphop/
      metal/
      pop/
```

Each genre folder contains 100 `.wav` files. The dataset is not included in this repository.

## Setup (Windows PowerShell)
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

## Run
Run everything with one command:
```powershell
python run_all.py
```

Or step by step:
```powershell
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_dcnn.py
python scripts/run_classical.py
```

Outputs are saved under `models/` and `results/`.

## Live demo
```powershell
python app.py
```

Open http://127.0.0.1:7860, pick a test song (or upload your own), and see the genre predicted by the CNN and by
each classical model.

## Notes
- All results above come from our own runs, not from the paper.
- The same train/test split is used for every feature set and classifier.
- Our random train/test split differs from the paper's, so exact numbers differ by a few points.
