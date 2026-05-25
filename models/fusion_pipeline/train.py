"""
LATE FUSION MULTIMODAL MODEL - TRAINING SCRIPT

CORRECT COMMAND TO RUN:
    python models/fusion_pipeline/train.py

Run from: Project Root Directory
Prerequisites: Train speech_pipeline and text_pipeline models first
This script trains a fusion head that combines audio and text predictions.
Results and model weights are saved to Results/ and models/fusion_pipeline/
"""

import os
import sys
import joblib
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==========================================
# PATHS & CONFIG
# ==========================================
BASE_DIR          = Path('.')
MANIFEST_PATH     = BASE_DIR / 'tess_manifest.csv'
MODELS_DIR        = BASE_DIR / 'models' / 'fusion_pipeline'
AUDIO_MODEL_PATH  = BASE_DIR / 'models' / 'speech_pipeline' / 'speech_only_model.pth'
TEXT_MODEL_PATH   = BASE_DIR / 'models' / 'text_pipeline' / 'tess_text_model_OAF.joblib'
RESULTS_METRICS   = BASE_DIR / 'Results' / 'metrics'
RESULTS_PLOTS     = BASE_DIR / 'Results' / 'plots'

EMOTION_ORDER = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. AUDIO BRANCH
# ==========================================
class SpeechEmotionModel(nn.Module):
    def __init__(self, input_size: int = 13, hidden_size: int = 64, num_classes: int = 7):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.dropout    = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)                       
        gru_out, _ = self.gru(x)                     
        temporal    = torch.mean(gru_out, dim=1)     
        logits      = self.classifier(self.dropout(temporal))
        return logits, temporal                      

# ==========================================
# 2. TEXT BRANCH HELPERS
# ==========================================
_NRC_EMOTIONS = ["anger","anticipation","disgust","fear","joy","negative","positive","sadness","surprise","trust"]

def _surface_features(word: str) -> list[float]:
    w = word.lower().strip()
    length      = len(w)
    vowel_ratio = sum(1 for c in w if c in "aeiou") / max(length, 1)
    syllables   = len(re.findall(r"[aeiou]+", w))
    return [float(length), vowel_ratio, float(syllables)]

def _vader_features(word: str, sid) -> list[float]:
    if sid is None:
        return [0.0, 0.0, 0.0, 0.0]
    s = sid.polarity_scores(word)
    return [s["compound"], s["pos"], s["neu"], s["neg"]]

def build_text_features(words: list[str], tfidf, nrc: dict, sid) -> np.ndarray:
    tfidf_mat = tfidf.transform(words).toarray()
    nrc_dim   = len(_NRC_EMOTIONS)
    nrc_mat   = np.array([nrc.get(w.lower(), [0.0]*nrc_dim) for w in words], dtype=np.float32)
    vader_mat = np.array([_vader_features(w, sid) for w in words], dtype=np.float32)
    surf_mat  = np.array([_surface_features(w) for w in words], dtype=np.float32)
    return np.concatenate([tfidf_mat, nrc_mat, vader_mat, surf_mat], axis=1)

def parse_transcript(path: str) -> str:
    stem  = Path(path).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return stem
    return "_".join(parts[1:-1]).replace("_", " ").strip()

# ==========================================
# 3. FUSION DATASET
# ==========================================
class TESSFusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder,
                 text_bundle: dict, sr: int = 22050,
                 max_len_sec: float = 3.0, n_mfcc: int = 13):
        self.df            = df.reset_index(drop=True)
        self.sr            = sr
        self.max_pad_len   = int(max_len_sec * sr)
        self.n_mfcc        = n_mfcc
        self.label_encoder = label_encoder
        self.labels        = label_encoder.transform(df['emotion'].values)

        words       = df['path'].apply(parse_transcript).tolist()
        feat_matrix = build_text_features(
            words, text_bundle['tfidf'], text_bundle['nrc'], text_bundle.get('sid')
        )
        text_clf        = text_bundle['model']
        self.text_probs = torch.tensor(
            text_clf.predict_proba(feat_matrix).astype(np.float32), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = self.labels[idx]

        y, sr = librosa.load(row['path'], sr=self.sr)
        y, _  = librosa.effects.trim(y, top_db=20)
        if len(y) < self.max_pad_len:
            y = np.pad(y, (0, self.max_pad_len - len(y)), mode='constant')
        else:
            y = y[:self.max_pad_len]
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)

        return (
            torch.tensor(mfcc, dtype=torch.float32),
            self.text_probs[idx],
            torch.tensor(label, dtype=torch.long),
        )

# ==========================================
# 4. FUSION HEAD & COMBINED MODEL
# ==========================================
class FusionHead(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, p_audio: torch.Tensor, p_text: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([p_audio, p_text], dim=1)
        return self.net(combined)

class TESSFusionModel(nn.Module):
    def __init__(self, audio_model: SpeechEmotionModel, fusion_head: FusionHead, num_classes: int = 7):
        super().__init__()
        self.audio_model = audio_model
        self.fusion_head = fusion_head
        self.num_classes = num_classes

    def forward(self, mfcc: torch.Tensor, text_probs: torch.Tensor):
        audio_logits, audio_embed = self.audio_model(mfcc)
        p_audio = F.softmax(audio_logits, dim=1)
        logits  = self.fusion_head(p_audio, text_probs)
        return logits, audio_embed

# ==========================================
# 5. MAIN TRAINING ROUTINE
# ==========================================
def train_fusion():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
    RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)
    df['emotion'] = df['emotion'].str.lower().str.strip().replace({'ps': 'surprise'})
    
    print(f"✓ Manifest loaded: {len(df)} samples")

    # === 60-20-20 STRATIFIED SPLIT ===
    train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['emotion'], random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df['emotion'], random_state=42)
    
    print(f"✓ Data Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} (Reserved)")

    le = LabelEncoder()
    le.fit(EMOTION_ORDER)
    
    # Load Text Model
    if not TEXT_MODEL_PATH.exists():
        print(f"❌ Text model not found at {TEXT_MODEL_PATH}")
        sys.exit(1)

    text_bundle = joblib.load(TEXT_MODEL_PATH)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        text_bundle['sid'] = SentimentIntensityAnalyzer()
    except ImportError:
        text_bundle['sid'] = None

    print("\nBuilding datasets (pre-computing text probabilities)...")
    train_dataset = TESSFusionDataset(train_df, le, text_bundle)
    val_dataset   = TESSFusionDataset(val_df,   le, text_bundle)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

    # Load Audio Model & Freeze it
    audio_model = SpeechEmotionModel().to(device)
    if AUDIO_MODEL_PATH.exists():
        audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
        print(f"✓ Loaded pre-trained audio model from {AUDIO_MODEL_PATH.name}")
    
    for param in audio_model.parameters():
        param.requires_grad = False
    print("✓ Audio backbone frozen — training fusion head only.")

    fusion_head  = FusionHead(num_classes=7).to(device)
    fusion_model = TESSFusionModel(audio_model, fusion_head).to(device)

    # Loss, Optimizer, Scheduler
    raw_weights   = compute_class_weight('balanced', classes=np.arange(7), y=train_dataset.labels)
    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, fusion_model.parameters()), lr=1e-3, weight_decay=0.01)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=15)

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience, epochs_no_improve = 5, 0
    num_epochs = 20

    print(f"\n📚 Training fusion head for {num_epochs} epochs (Early Stop Patience={patience})...")

    for epoch in range(1, num_epochs + 1):
        fusion_model.train()
        run_loss, correct, total = 0.0, 0, 0

        for mfcc, text_probs, labels in train_loader:
            mfcc, text_probs, labels = mfcc.to(device), text_probs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = fusion_model(mfcc, text_probs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * mfcc.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        train_loss = run_loss / total
        train_acc  = correct / total * 100

        fusion_model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for mfcc, text_probs, labels in val_loader:
                mfcc, text_probs, labels = mfcc.to(device), text_probs.to(device), labels.to(device)
                logits, _  = fusion_model(mfcc, text_probs)
                loss       = criterion(logits, labels)
                v_loss    += loss.item() * mfcc.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total * 100
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{num_epochs}] -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(fusion_model.state_dict(), MODELS_DIR / 'tess_fusion_model.pth')
            print(f"  --> Best model saved (Val Acc: {best_val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n⏹️ Early stopping triggered at epoch {epoch}")
                break

    # Save artifacts
    joblib.dump(le, MODELS_DIR / 'fusion_label_encoder.pkl')
    pd.DataFrame(history).to_csv(RESULTS_METRICS / 'tess_fusion_metrics_history.csv', index=False)

    # === DUAL TRAINING PLOT ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['epoch'], history['train_loss'], 'r-o', label='Train Loss')
    axes[0].plot(history['epoch'], history['val_loss'], 'orange', marker='x', label='Val Loss')
    axes[0].set_title('Fusion Model Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history['epoch'], history['train_acc'], 'b-o', label='Train Acc')
    axes[1].plot(history['epoch'], history['val_acc'], 'g-s', label='Val Acc')
    axes[1].set_title('Fusion Model Training vs Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_PLOTS / 'tess_fusion_training_history.png', dpi=150)
    plt.close()

    print("\n✅ Fusion training complete! Weights, Metrics CSV, and Dual-Plots saved.")

if __name__ == "__main__":
    train_fusion()