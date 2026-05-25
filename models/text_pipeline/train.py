"""
TESS TEXT-ONLY BASELINE - TRAINING SCRIPT (TF-IDF + Logistic Regression)

CORRECT COMMAND TO RUN:
    python models/text_pipeline/train.py

Run from: Project Root Directory
This script trains the TF-IDF + Logistic Regression model on TESS transcripts.
Results and model weights are saved to Results/ and models/text_pipeline/
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = BASE_DIR / "tess_manifest.csv"
MODELS_DIR = BASE_DIR / "models" / "text_pipeline"
RESULTS_METRICS_DIR = BASE_DIR / "Results" / "metrics"
RESULTS_PLOTS_DIR = BASE_DIR / "Results" / "plots"

EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ---------------------------------------------------------------------------
# Optional NRC lexicon helper
# ---------------------------------------------------------------------------
_NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "negative", "positive", "sadness", "surprise", "trust",
]

def _load_nrc(nrc_path: Path | None) -> dict[str, list[float]]:
    if nrc_path is None or not nrc_path.exists():
        return {}
    lexicon: dict[str, list[float]] = {}
    with nrc_path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, flag = parts
            if emotion not in _NRC_EMOTIONS:
                continue
            vec = lexicon.setdefault(word.lower(), [0.0] * len(_NRC_EMOTIONS))
            idx = _NRC_EMOTIONS.index(emotion)
            vec[idx] = float(flag)
    return lexicon

# ---------------------------------------------------------------------------
# Feature engineering
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
    tfidf: TfidfVectorizer | None,
    nrc: dict[str, list[float]],
    sid,
    fit_tfidf: bool = False,
) -> np.ndarray:
    corpus = words 
    if fit_tfidf:
        tfidf_mat = tfidf.fit_transform(corpus).toarray()
    else:
        tfidf_mat = tfidf.transform(corpus).toarray()

    nrc_dim = len(_NRC_EMOTIONS)
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
def plot_training_history(history: dict, save_path: Path, train_speaker: str, test_speaker: str) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(epochs, history["train_accuracy"], "b-o", label="Train Acc")
    ax.plot(epochs, history["dev_accuracy"],   "g-s", label="Dev Acc")
    ax.set_title(f"TESS Text Training Accuracy\n(Train={train_speaker} → Test={test_speaker})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    loss_proxy = [1.0 - a for a in history["train_accuracy"]]
    ax2.plot(epochs, loss_proxy, "r-o", label="Train Loss (proxy)")
    ax2.set_title("TESS Text Training Loss (proxy)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Proxy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved training history plot → {save_path.name}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a text-only model on TESS transcripts.")
    parser.add_argument("--train-speaker", default="OAF", choices=["OAF", "YAF"])
    parser.add_argument("--nrc-path", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        print("  VADER loaded ✓")
    except ImportError:
        sid = None
        print("  VADER not installed — VADER features will be zeros.")

    nrc = _load_nrc(Path(args.nrc_path) if args.nrc_path else None)
    if nrc:
        print(f"  NRC lexicon loaded ({len(nrc)} entries) ✓")
    else:
        print("  NRC lexicon not found — NRC features will be zeros.")

    df = load_tess_dataframe()
    train_speaker = args.train_speaker
    test_speaker = "YAF" if train_speaker == "OAF" else "OAF"
    
    print(f"\n  Train speaker : {train_speaker}")
    print(f"  Total samples : {len(df)}")

    train_full = df[df["speaker"] == train_speaker].reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_idx, dev_idx = next(sss.split(train_full, train_full["emotion"]))
    train_df = train_full.iloc[train_idx].reset_index(drop=True)
    dev_df = train_full.iloc[dev_idx].reset_index(drop=True)

    print(f"  Train samples : {len(train_df)}")
    print(f"  Dev samples   : {len(dev_df)}")

    le = LabelEncoder()
    all_emotions = sorted(df["emotion"].unique().tolist())
    le.fit(all_emotions)

    y_train = le.transform(train_df["emotion"].values)
    y_dev = le.transform(dev_df["emotion"].values)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1, sublinear_tf=True)

    print("\n  Building feature matrices …")
    x_train = build_feature_matrix(train_df["transcript"].tolist(), tfidf, nrc, sid, fit_tfidf=True)
    x_dev = build_feature_matrix(dev_df["transcript"].tolist(), tfidf, nrc, sid)
    print(f"  Feature dimensionality: {x_train.shape[1]}")

    classes = np.arange(len(all_emotions))
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_map = {i: w for i, w in enumerate(class_weights)}

    history = {"epoch": [], "train_accuracy": [], "dev_accuracy": []}
    best_dev_acc = -1.0
    best_model = None

    print(f"\n  Training for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        # FIX: Swapped LinearSVC for LogisticRegression to natively handle probabilities
        clf = LogisticRegression(class_weight=class_weight_map, max_iter=200 * epoch, C=0.5, random_state=42, solver='lbfgs')
        clf.fit(x_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(x_train))
        dev_acc = accuracy_score(y_dev, clf.predict(x_dev))

        history["epoch"].append(epoch)
        history["train_accuracy"].append(train_acc)
        history["dev_accuracy"].append(dev_acc)

        print(f"  Epoch [{epoch:02d}/{args.epochs}] Train: {train_acc:.4f}  Dev: {dev_acc:.4f}")

        # Save the best model natively without breaking calibrators
        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            best_model = clf   

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"tess_text_model_{train_speaker}.joblib"
    joblib.dump(
        {
            "model": best_model,
            "tfidf": tfidf,
            "label_encoder": le,
            "nrc": nrc,
            "train_speaker": train_speaker,
            "test_speaker": test_speaker,
            "feature_dim": x_train.shape[1],
            "all_emotions": all_emotions,
        },
        model_path,
    )
    print(f"\n  Saved model → {model_path.name}")

    history_path = RESULTS_METRICS_DIR / "tess_text_metrics_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"  Saved metrics history → {history_path.name}")

    plot_training_history(
        history,
        save_path=RESULTS_PLOTS_DIR / "tess_text_training_history.png",
        train_speaker=train_speaker,
        test_speaker=test_speaker,
    )

    print("\n  ✓ Training outputs saved successfully.\n")

if __name__ == "__main__":
    main()