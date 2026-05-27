"""
TESS TEXT-ONLY BASELINE - TESTING SCRIPT (TF-IDF + Logistic Regression)

CORRECT COMMAND TO RUN:
    python models/text_pipeline/test.py

Run from: Project Root Directory
Prerequisite: Train model first using train.py
This script evaluates the TF-IDF + Logistic Regression model on TESS dataset.
Generates confusion matrix, F1 scores, and performance metrics.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = BASE_DIR / "tess_manifest.csv"
RESULTS_METRICS_DIR = BASE_DIR / "Results" / "metrics"
RESULTS_PLOTS_DIR = BASE_DIR / "Results" / "plots"

_NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "negative", "positive", "sadness", "surprise", "trust",
]

# ---------------------------------------------------------------------------
# Feature engineering (Must match train.py exactly)
# ---------------------------------------------------------------------------
def _surface_features(word: str) -> list[float]:
    w = word.lower().strip()
    length = len(w)
    vowels = sum(1 for c in w if c in "aeiou")
    vowel_ratio = vowels / max(length, 1)
    syllables = len(re.findall(r"[aeiou]+", w))
    return [float(length), vowel_ratio, float(syllables)]

def _vader_features(word: str, sid) -> list[float]:
    if sid is None:
        return [0.0, 0.0, 0.0, 0.0]
    scores = sid.polarity_scores(word)
    return [scores["compound"], scores["pos"], scores["neu"], scores["neg"]]

def build_feature_matrix(
    words: list[str],
    tfidf,
    nrc: dict[str, list[float]],
    sid,
) -> np.ndarray:
    nrc_dim = len(_NRC_EMOTIONS)
    
    # Notice we use .transform() instead of .fit_transform() here!
    tfidf_mat = tfidf.transform(words).toarray()
    
    nrc_mat = np.array([nrc.get(w.lower(), [0.0] * nrc_dim) for w in words], dtype=np.float32)
    vader_mat = np.array([_vader_features(w, sid) for w in words], dtype=np.float32)
    surf_mat = np.array([_surface_features(w) for w in words], dtype=np.float32)

    return np.concatenate([tfidf_mat, nrc_mat, vader_mat, surf_mat], axis=1)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def parse_transcript(path: str) -> str:
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return stem
    word = "_".join(parts[1:-1])
    return word.replace("_", " ").strip()

def load_tess_dataframe() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"tess_manifest.csv not found at {MANIFEST_PATH}. Run data_loader.py first.")
    df = pd.read_csv(MANIFEST_PATH)
    df["emotion"] = df["emotion"].str.lower().str.strip()
    df["emotion"] = df["emotion"].replace({"ps": "surprise"})
    df["speaker"] = df["path"].str.extract(r"([OY]AF)")[0]
    df["transcript"] = df["path"].apply(parse_transcript)
    return df

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_training_history(history_df: pd.DataFrame, save_path: Path, train_speaker: str, test_speaker: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history_df["epoch"], history_df["train_accuracy"], "b-o", label="Train Acc")
    if "dev_accuracy" in history_df.columns:
        ax.plot(history_df["epoch"], history_df["dev_accuracy"], "g-s", label="Dev Acc")
    ax.set_title(f"TESS Text Training Accuracy\n(Train={train_speaker} → Test={test_speaker})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    loss_proxy = 1.0 - history_df["train_accuracy"]
    ax2.plot(history_df["epoch"], loss_proxy, "r-o", label="Train Loss (proxy)")
    ax2.set_title("TESS Text Training Loss (proxy)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Proxy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved training history plot → {save_path.name}")

def plot_confusion_matrix(y_true, y_pred, labels: list[str], save_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {save_path.name}")

def plot_f1_per_class(y_true, y_pred, labels: list[str], save_path: Path, title: str) -> None:
    f1_scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    colors = ["#e74c3c" if s < 0.35 else "#f39c12" if s < 0.55 else "#2ecc71" for s in f1_scores]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, f1_scores, color=colors, edgecolor="white", linewidth=0.8)
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{score:.2f}", ha="center", va="bottom", fontsize=10)
                 
    plt.title(title)
    plt.xlabel("Emotion")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved per-class F1 plot → {save_path.name}")

def plot_tsne(features: np.ndarray, labels: list[str], emotion_names: list[str], save_path: Path, title: str) -> None:
    print(f"  Generating t-SNE plot (this may take a moment)...")
    # Using a small perplexity because cluster sizes might be small
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=min(30, len(labels)-1))
    x_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=x_2d[:, 0], y=x_2d[:, 1],
        hue=labels,
        hue_order=emotion_names,
        palette="husl",
        alpha=0.8,
        s=60,
        edgecolor="w",
        linewidth=0.5
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Emotions")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved t-SNE graph → {save_path.name}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained TESS text-only model checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=str(Path("models") / "text_pipeline" / "tess_text_model_OAF.joblib"),
        help="Path to the saved .joblib checkpoint (relative to project root).",
    )
    args = parser.parse_args()

    checkpoint_path = BASE_DIR / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}\nRun train.py first.")

    print(f"\n  Loading checkpoint: {checkpoint_path.name}")
    ckpt = joblib.load(checkpoint_path)

    model = ckpt["model"]
    tfidf = ckpt.get("tfidf") 
    le = ckpt.get("label_encoder")
    nrc = ckpt.get("nrc", {})
    train_speaker = ckpt.get("train_speaker", "OAF")
    test_speaker = ckpt.get("test_speaker", "YAF" if train_speaker == "OAF" else "OAF")
    all_emotions = ckpt.get("all_emotions")

    if tfidf is None:
        raise ValueError("Checkpoint missing TF-IDF vectorizer. Retrain with train.py.")

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        print("  VADER loaded ✓")
    except ImportError:
        sid = None
        print("  VADER not installed — VADER features will be zeros.")

    df = load_tess_dataframe()
    test_df = df[df["speaker"] == test_speaker].reset_index(drop=True)
    
    print(f"  Train speaker : {train_speaker}")
    print(f"  Test speaker  : {test_speaker}")
    print(f"  Test samples  : {len(test_df)}")

    # Build features using the exact same logic as training
    x_test = build_feature_matrix(test_df["transcript"].tolist(), tfidf, nrc, sid)

    # Make predictions
    preds_enc = model.predict(x_test)
    pred_labels = le.inverse_transform(preds_enc).tolist()
    true_labels = test_df["emotion"].tolist()

    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, zero_division=0)
    present_labels = all_emotions if all_emotions else sorted(df["emotion"].unique().tolist())

    print(f"\n  ═══ Test Accuracy : {accuracy:.4f} ═══")
    print(report)

    # Save metrics and plots
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    perf_path = RESULTS_METRICS_DIR / "tess_text_only_performance.txt"
    with perf_path.open("w", encoding="utf-8") as fh:
        fh.write("=== TESS TEXT-ONLY EVALUATION RESULTS ===\n")
        fh.write(f"Checkpoint    : {checkpoint_path.name}\n")
        fh.write(f"Train speaker : {train_speaker}\n")
        fh.write(f"Test speaker  : {test_speaker}\n")
        fh.write(f"Test accuracy : {accuracy:.4f}\n\n")
        fh.write(report)
    print(f"\n  Saved performance report → {perf_path.name}")

    # Reload history to draw the dual training/dev plot
    history_path = RESULTS_METRICS_DIR / "tess_text_metrics_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        plot_training_history(
            history_df,
            save_path=RESULTS_PLOTS_DIR / "tess_text_training_history.png",
            train_speaker=train_speaker,
            test_speaker=test_speaker,
        )

    plot_confusion_matrix(
        true_labels,
        pred_labels,
        labels=present_labels,
        save_path=RESULTS_PLOTS_DIR / "tess_text_confusion_matrix.png",
        title=f"TESS Text — Confusion Matrix\n(Train={train_speaker} → Test={test_speaker}  Acc={accuracy:.4f})",
    )

    plot_f1_per_class(
        true_labels,
        pred_labels,
        labels=present_labels,
        save_path=RESULTS_PLOTS_DIR / "tess_text_f1_per_class.png",
        title=f"TESS Text — Per-class F1 Score\n(Train={train_speaker} → Test={test_speaker})",
    )

    # -----------------------------------------------------------------------
    # Generate t-SNE Plot in Results/plots
    # -----------------------------------------------------------------------
    plot_tsne(
        x_test,
        true_labels,
        emotion_names=present_labels,
        save_path=RESULTS_PLOTS_DIR / "tess_text_tsne_visualization.png",
        title=f"TESS Text Features — t-SNE Visualization\n(Test Data: {test_speaker} Speaker)",
    )

    print("\n  ✓ All test outputs saved successfully.\n")

if __name__ == "__main__":
    main()