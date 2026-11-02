"""Build report/writeup.pdf (two-page project summary) from the files in results/.

Edit TEAM below, then run:  python report/build_writeup.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

TEAM = ("Pranav Prabhakar (PES1UG24AM463), Haneesh E (PES1UG24AM471) &nbsp;|&nbsp; "
        "UE24CS352A Machine Learning, PES University")
REPO = "github.com/pranav-prabhakar-dev/music-genre-dcnn-classical"
OUT = ROOT / "report" / "writeup.pdf"

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.2,
                      leading=11.6, alignment=TA_JUSTIFY, spaceAfter=3)
H = ParagraphStyle("h", parent=BODY, fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                   spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1F3A5F"))
TITLE = ParagraphStyle("t", parent=BODY, fontName="Helvetica-Bold", fontSize=14.5, leading=18,
                       alignment=TA_CENTER, spaceAfter=2)
SUB = ParagraphStyle("s", parent=BODY, fontSize=8.8, alignment=TA_CENTER, spaceAfter=6,
                     textColor=colors.HexColor("#444444"))
CAP = ParagraphStyle("c", parent=BODY, fontSize=7.8, leading=9.5, alignment=TA_CENTER,
                     textColor=colors.HexColor("#444444"), spaceAfter=4)
BUL = ParagraphStyle("b", parent=BODY, leftIndent=10, bulletIndent=2, spaceAfter=1.5)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def results_table():
    res = pd.read_csv(C.RESULTS / "classical_results.csv")
    dcnn = json.loads((C.RESULTS / "dcnn_metrics.json").read_text())
    seeds = pd.read_csv(C.RESULTS / "seed_study_summary.csv")
    algs = ["LR", "GDA", "RF", "SVM"]
    header = ["Input features (dim)", "split"] + algs + ["5-seed test mean ± std (LR / GDA / RF / SVM)"]
    data = [header]
    best = res["test_acc"].max()
    for feat in res["features"].unique():
        sub = res[res.features == feat].set_index("algorithm")
        seed_txt = "– (does not depend on the DCNN seed)"
        layer = "DCNN layer 1" if "layer 1" in feat else "DCNN layer 2" if "layer 2" in feat else None
        if layer:
            s = seeds[seeds.features == layer].set_index("algorithm")
            seed_txt = " / ".join(f"{s.loc[a, 'mean']:.2f}±{s.loc[a, 'std']:.2f}" for a in algs)
        # spanned cells show the content of their top-left cell, so seed_txt goes on the train row
        data.append([feat, "train"] + [f"{sub.loc[a, 'train_acc']:.2f}" for a in algs] + [seed_txt])
        data.append(["", "test"] + [f"<b>{v:.2f}</b>" if v == best else f"{v:.2f}"
                                    for v in (sub.loc[a, "test_acc"] for a in algs)] + [""])
    sd = seeds[seeds.algorithm == "DCNN"].iloc[0]
    data.append(["DCNN alone (5,973 params)", "train / test",
                 f"{dcnn['train_acc']:.2f} / <b>{dcnn['test_acc']:.2f}</b>", "", "", "",
                 f"{sd['mean']:.2f} ± {sd['std']:.2f}"])
    cell = ParagraphStyle("cell", parent=BODY, fontSize=7.8, leading=9, alignment=TA_CENTER, spaceAfter=0)
    data = [[Paragraph(str(x), cell) for x in row] for row in data]
    t = Table(data, colWidths=[3.6 * cm, 1.5 * cm] + [1.25 * cm] * 4 + [5.6 * cm])
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9AA5B1")),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
             ("SPAN", (2, len(data) - 1), (5, len(data) - 1))]
    for r in range(1, len(data) - 1, 2):
        style += [("SPAN", (0, r), (0, r + 1)), ("SPAN", (6, r), (6, r + 1))]
        if (r // 2) % 2:
            style.append(("BACKGROUND", (0, r), (-1, r + 1), colors.HexColor("#F5F7FA")))
    t.setStyle(TableStyle(style))
    return t


def ablation_table():
    df = pd.read_csv(C.RESULTS / "dcnn_ablation.csv")
    cell = ParagraphStyle("acell", parent=BODY, fontSize=7.8, leading=9, spaceAfter=0)
    rows = [[Paragraph(f"<b>{h}</b>", cell) for h in
             ("DCNN layers: filters, kernel, dilation, pool (d = dropout)", "Train", "Test (3 seeds)")]]
    for _, r in df.iterrows():
        rows.append([Paragraph(r.config.replace("|", "&nbsp;|&nbsp;"), cell),
                     Paragraph(f"{r.train_acc:.2f}", cell),
                     Paragraph(f"{r.test_acc_mean:.2f} ± {r.test_acc_std:.2f}", cell)])
    t = Table(rows, colWidths=[6.2 * cm, 1.2 * cm, 2.0 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9AA5B1")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    return t


def fig(name, width):
    path = C.FIGURES / name
    img = Image(str(path))
    img.drawWidth, img.drawHeight = width, width * img.imageHeight / img.imageWidth
    return img


def build():
    res = pd.read_csv(C.RESULTS / "classical_results.csv").set_index(["features", "algorithm"])["test_acc"]
    svm = pd.read_csv(C.RESULTS / "svm_gamma_check.csv")
    auto = svm[svm.gamma == "auto"].set_index("features")
    dcnn = json.loads((C.RESULTS / "dcnn_metrics.json").read_text())
    seeds = pd.read_csv(C.RESULTS / "seed_study_summary.csv").set_index(["features", "algorithm"])
    raw, l1, l2 = "Raw MFCC (25800)", "DCNN layer 1 (320)", "DCNN layer 2 (48)"

    story = [
        P("Combining a Dilated CNN with Classical Algorithms for Music Genre Classification", TITLE),
        P(TEAM, SUB),

        P("1. Problem statement", H),
        P("Automatic music genre classification is needed to organise large audio collections and "
          "power recommendation and playlist systems. Genres are loosely defined and audio is "
          "high-dimensional (one 30 s clip is ~660k samples), which makes this hard for classical models "
          "trained on hand-crafted inputs. Following Li, Xue &amp; Zhang (CS229, 2018), we ask: <i>can a "
          "small Dilated CNN, used as a learned feature extractor, make classical classifiers (Softmax "
          "Logistic Regression, GDA, Random Forest, SVM) more accurate, possibly more accurate than the "
          "CNN itself?</i>"),

        P("2. Dataset", H),
        P("We use <b>GTZAN</b> (Tzanetakis &amp; Cook, 2002): 1000 30-second, 22.05 kHz mono .wav clips "
          "in 10 genres. As in the paper we keep 5 genres (<b>classical, hip-hop, metal, pop, blues</b>), "
          "giving 500 clips (100 per genre). We use one stratified 80/20 split (400 train / 100 test, "
          "seed 42) for every experiment. Each clip becomes a <b>20 × 1290 MFCC matrix</b> (librosa, "
          "n_fft 2048, hop 512, cropped/tiled to 1290 frames), i.e. 25,800 values when flattened."),

        P("3. Approach", H),
        *bullets([
            "<b>Baselines:</b> the flattened MFCC vector (25,800-d), and its top-50 PCA components, fed "
            "to LR (multinomial), GDA (shared covariance = LDA), RF (300 Gini trees, max depth 7) and an "
            "RBF-SVM (one-vs-one, γ = 1/(d·Var X)). All inputs are standardised with training statistics.",
            "<b>Dilated CNN:</b> the paper's final 2-unit 1D architecture over time, with the 20 MFCCs as "
            "channels. Unit 1: Conv1d(8 filters, k = 16, dilation 8) → ReLU → AvgPool 32 → Dropout 0.5 → "
            "BatchNorm. Unit 2: Conv1d(24, k = 16, dilation 2) → ReLU → AvgPool 4 → Dropout → BatchNorm. "
            "Then a dense softmax layer. Dilation lets each unit-1 filter span 121 frames (≈2.8 s) "
            "with only 16 taps; the whole network has 5,973 parameters. Trained with Adam (lr 10<super>-3</super>), cross-entropy, "
            "batch 32, for a fixed 150 epochs. The test set is never used for model selection.",
            "<b>Combination:</b> the trained DCNN's unit-1 (8 × 40 = <b>320-d</b>) and unit-2 "
            "(24 × 2 = <b>48-d</b>) activations become the inputs to the four classical models, "
            "giving 4 feature sets × 4 algorithms. Because one 100-clip test set is noisy (±3 points), "
            "we also repeat DCNN training plus the classical models over 5 seeds."]),

        P("4. Implementation", H),
        P("Python 3.13 with <b>librosa</b> (features), <b>PyTorch</b> (DCNN), <b>scikit-learn</b> "
          "(classical pipelines, PCA), matplotlib/seaborn (figures) and a <b>Gradio</b> web app for the "
          "live demo (pick a held-out clip or upload/record audio and see the DCNN probabilities next to "
          "all 8 classical predictions). The code is modular (<font face='Courier'>src/</font> library, "
          "<font face='Courier'>scripts/</font> for each stage, <font face='Courier'>run_all.py</font> "
          "end to end) and the full pipeline runs on a laptop CPU in about 5 minutes. Repository: " + REPO + "."),

        P("5. Results", H),
        KeepTogether([results_table(),
                      P("Table 1: Train/test accuracy of each classical algorithm on each input "
                        "(seed 42), plus the 5-seed mean ± std test accuracy for DCNN features. "
                        "Best single-run test accuracy in bold.", CAP)]),
        Table([[fig("classical_results_bars.png", 9.0 * cm), fig("pca_features_3d.png", 9.0 * cm)]],
              colWidths=[9.2 * cm, 9.2 * cm]),
        P("Figure 1: Left, test accuracy by algorithm and input. Right, 3D PCA of the test clips in "
          "each feature space. Genres separate more cleanly in the DCNN layers than in the raw MFCCs.", CAP),
        *bullets([
            f"<b>DCNN features help almost everywhere.</b> RF improves from {res[(raw, 'RF')]:.2f} to "
            f"{res[(l1, 'RF')]:.2f}, SVM from {res[(raw, 'SVM')]:.2f} to {res[(l1, 'SVM')]:.2f}, and LR from "
            f"{res[(raw, 'LR')]:.2f} to {res[(l2, 'LR')]:.2f}. The 48-d layer-2 code beats the 50-d PCA "
            "code for every algorithm, so the CNN learns a much more discriminative low-dimensional "
            "representation than linear variance-based compression.",
            f"<b>Classical model beats the CNN.</b> Over 5 seeds the RBF-SVM on layer-1 features reaches "
            f"{seeds.loc[('DCNN layer 1', 'SVM'), 'mean']:.3f} ± {seeds.loc[('DCNN layer 1', 'SVM'), 'std']:.3f}, "
            f"above the DCNN's own softmax ({seeds.loc[('DCNN alone', 'DCNN'), 'mean']:.3f} ± "
            f"{seeds.loc[('DCNN alone', 'DCNN'), 'std']:.3f}). This agrees with the paper: a max-margin "
            "classifier on the richer, earlier representation acts as a regulariser on top of a small, "
            "imperfect network.",
            f"<b>GDA fails on layer 1</b> ({res[(l1, 'GDA')]:.2f}, the same as the paper's 0.61). One shared "
            "Gaussian covariance over 320 correlated, pooled ReLU activations estimated from 400 clips is a "
            f"poor model. On the compact layer-2 code GDA recovers to {res[(l2, 'GDA')]:.2f}.",
            f"<b>We explain the paper's SVM anomaly.</b> The paper reports SVM test accuracy of 0.28 / 0.21 "
            f"on raw / PCA input. We reproduce this pattern (train 1.00, test "
            f"{auto.loc[raw, 'test_acc']:.2f} / {auto.loc['PCA of raw (50)', 'test_acc']:.2f}) using "
            "<i>unscaled</i> MFCCs with γ = 1/d, scikit-learn's pre-2019 default. The kernel becomes too "
            "narrow and the SVM memorises the training set. With standardised features the same SVM scores "
            f"{res[(raw, 'SVM')]:.2f} / {res[('PCA of raw (50)', 'SVM')]:.2f}.",
            f"The DCNN alone reaches {dcnn['test_acc']:.2f} test accuracy (paper: 0.84). Most errors are "
            "between metal, blues and hip-hop. Classical music is recognised almost perfectly (Fig. 2). "
            "In the architecture study (Table 2) a single first-layer filter is far too few (0.64; paper "
            "0.49). Every other configuration lands at 0.85–0.91, within about two standard deviations. "
            "Unlike the paper, larger filters did not help and a third unit did not overfit, so on this "
            "data the architecture choice matters less than the paper suggests."]),
        Spacer(1, 4),
        Table([[fig("confusion_dcnn.png", 6.4 * cm), ablation_table()]],
              colWidths=[7.4 * cm, 10.8 * cm], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
        P("Figure 2 (left): DCNN confusion matrix on the 100 test clips. Table 2 (right): DCNN architecture "
          "study (paper Tables V.1/V.2), mean ± std test accuracy over 3 seeds.", CAP),

        P("6. Conclusions", H),
        P("A small dilated CNN trained on only 400 clips is an effective feature extractor. Its activations "
          "raise the accuracy of all four classical algorithms (except GDA on the wide layer), and an SVM "
          "on those activations matches or beats the network itself, as the original paper claims. "
          "Feature preprocessing and kernel hyper-parameters matter as much as the model: the paper's "
          "dramatic SVM failure on raw features was an artefact of unscaled inputs and γ. With a 100-clip "
          "test set, differences of 1–3 points are within seed noise, so we report multi-seed averages. "
          "<b>Future work:</b> recurrent (LSTM/GRU) or attention feature extractors for the sequential "
          "structure, all 10 GTZAN genres with an artist-filtered split, data augmentation by "
          "taking several 3–5 s windows per clip, and cross-validated hyper-parameter search for the "
          "classical models."),
    ]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                            title="DCNN + Classical Algorithms for Music Genre Classification")
    doc.build(story)
    print("Wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    build()
