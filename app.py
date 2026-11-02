"""Live demo: classify a music clip with the DCNN and with classical models on DCNN features.

Run:  python app.py   (then open the printed http://127.0.0.1:7860 link)
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd
import torch

from src import config as C
from src.classical import ALGORITHMS
from src.dcnn import extract_features, load_checkpoint
from src.features import load_dataset, mfcc_from_file

MODEL, SCALER, META = load_checkpoint()
CLASSICAL = {(alg, tag): joblib.load(C.MODELS / f"{alg.lower()}_{tag}.joblib")
             for alg in ALGORITHMS for tag in ("l1", "l2")}
TEST_FILES = list(load_dataset()["files_test"])


def mfcc_figure(mfcc, title):
    fig, ax = plt.subplots(figsize=(8, 2.6))
    im = ax.imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Time frame"); ax.set_ylabel("MFCC #"); ax.set_title(title)
    fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    return fig


def classify(test_clip, uploaded, offset):
    if uploaded:
        path, true_genre, offset = uploaded, "unknown (uploaded)", float(offset)
    elif test_clip:
        path, true_genre, offset = C.DATA_RAW / test_clip, test_clip.split("/")[0], 0.0
    else:
        raise gr.Error("Pick a test-set clip or upload / record an audio file.")

    mfcc = mfcc_from_file(path, offset=offset)
    X = mfcc[None]
    with torch.no_grad():
        probs = torch.softmax(MODEL(torch.from_numpy(SCALER.transform(X))), 1).numpy()[0]
    dcnn = {g: float(p) for g, p in zip(C.GENRES, probs)}

    l1, l2 = extract_features(MODEL, SCALER, X)
    rows = []
    for alg in ALGORITHMS:
        for tag, F in (("l1", l1), ("l2", l2)):
            p = CLASSICAL[(alg, tag)].predict_proba(F)[0]
            rows.append({"model": f"{alg} on DCNN layer {tag[1]}",
                         "prediction": C.GENRES[int(p.argmax())],
                         "confidence": round(float(p.max()), 3)})
    table = pd.DataFrame(rows)
    votes = table["prediction"].value_counts()
    summary = (f"**True genre:** {true_genre}  \n**DCNN:** {C.GENRES[int(probs.argmax())]} "
               f"({probs.max():.0%})  \n**Classical models (majority of 8):** {votes.index[0]} "
               f"({votes.iloc[0]}/8 votes)")
    return dcnn, table, mfcc_figure(mfcc, f"MFCC input (20 x {C.N_FRAMES})"), summary


with gr.Blocks(title="Music Genre Classification - DCNN + Classical") as demo:
    gr.Markdown("# Music genre classification: Dilated CNN + classical algorithms\n"
                f"Genres: {', '.join(C.GENRES)}. DCNN test accuracy: **{META.get('test_acc', 0):.2f}**. "
                "Choose a held-out GTZAN test clip, or upload/record your own audio (30 s are used).")
    with gr.Row():
        with gr.Column(scale=1):
            clip = gr.Dropdown(TEST_FILES, label="Held-out test clip (never seen in training)",
                               value=TEST_FILES[0])
            audio = gr.Audio(label="...or upload / record audio", type="filepath",
                             sources=["upload", "microphone"])
            offset = gr.Slider(0, 120, value=0, step=5, label="Start offset for uploaded audio (s)")
            btn = gr.Button("Classify", variant="primary")
            summary = gr.Markdown()
        with gr.Column(scale=2):
            dcnn_out = gr.Label(num_top_classes=5, label="DCNN genre probabilities")
            table = gr.Dataframe(label="Classical algorithms on DCNN features", interactive=False)
    plot = gr.Plot(label="MFCC features")
    btn.click(classify, [clip, audio, offset], [dcnn_out, table, plot, summary])

if __name__ == "__main__":
    demo.launch()
