import os
import sys
import json
import joblib
 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
 
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVED_MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
if not os.path.exists(os.path.join(BASE_DIR, "tess_manifest.csv")):
    print("\n❌ ERROR: Run from project root:  python archived_models/text_pipeline/train_bigru.py")
    sys.exit(1)
 
TEXT_TRAIN_CSV   = os.path.join(BASE_DIR, "data", "MELD", "train_sent_emo_balanced.csv")
TEXT_DEV_CSV     = os.path.join(BASE_DIR, "data", "MELD", "dev_sent_emo.csv")
MODELS_DIR       = os.path.join(ARCHIVED_MODELS_DIR, "text_pipeline")
TEXT_VOCAB_PATH  = os.path.join(MODELS_DIR, "text_vocab.json")
TEXT_MODEL_PATH  = os.path.join(MODELS_DIR, "text_bigru_model.pth")
LABEL_ENC_PATH   = os.path.join(MODELS_DIR, "label_encoder_bigru.pkl")
RESULTS_METRICS  = os.path.join(ARCHIVED_MODELS_DIR, "Results", "metrics")
RESULTS_PLOTS    = os.path.join(ARCHIVED_MODELS_DIR, "Results", "plots")
 
VALID_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
LABEL_MAP      = {"anger": "angry", "joy": "happy", "sadness": "sad"}
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 
 
# ── Dataset ───────────────────────────────────────────────────────────────────
class TextEmotionDataset(Dataset):
    """
    BiGRU text emotion dataset with vocabulary management.
    
    Parameters
    ----------
    label_encoder : LabelEncoder or None
        Pass a pre-fit encoder (from training) for dev/test splits.
        Pass None to fit a new encoder (training only).
    build_vocab : bool
        Whether to build vocab from this dataset (training=True only).
    """
    def __init__(self, csv_file: str, is_training: bool = True, max_length: int = 50,
                 label_encoder: LabelEncoder | None = None, vocab: dict | None = None,
                 build_vocab: bool = True):
        df = pd.read_csv(csv_file)
 
        # Normalise column names
        if 'Utterance' in df.columns and 'text' not in df.columns:
            df['text'] = df['Utterance']
        if 'Emotion' in df.columns and 'emotion' not in df.columns:
            df['emotion'] = df['Emotion']
 
        df['emotion'] = df['emotion'].str.lower().replace(LABEL_MAP)
        df = df[df['emotion'].isin(VALID_EMOTIONS)].reset_index(drop=True)
 
        self.texts = df['text'].astype(str).values
        self.max_length = max_length
 
        # FIX: use a shared encoder — never refit on test/dev
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(VALID_EMOTIONS)
        else:
            self.label_encoder = label_encoder
 
        self.labels = self.label_encoder.transform(df['emotion'].values)
 
        # Vocabulary management
        self.PAD_TOKEN = 0
        self.UNK_TOKEN = 1
 
        if build_vocab and vocab is None:
            self.vocab = {"<PAD>": self.PAD_TOKEN, "<UNK>": self.UNK_TOKEN}
            self._build_vocab()
        else:
            self.vocab = vocab if vocab is not None else {}
 
    def _build_vocab(self):
        """Build vocabulary from training texts."""
        import re
        word_idx = 2
        for text in self.texts:
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            words = clean_text.split()
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = word_idx
                    word_idx += 1
 
    def __len__(self) -> int:
        return len(self.texts)
 
    def __getitem__(self, idx):
        import re
        text = self.texts[idx]
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
 
        word_ids = [self.vocab.get(word, self.UNK_TOKEN) for word in words]
 
        if len(word_ids) < self.max_length:
            word_ids = word_ids + [self.PAD_TOKEN] * (self.max_length - len(word_ids))
        else:
            word_ids = word_ids[:self.max_length]
 
        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
 
 
# ── Model ─────────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    """Attention mechanism over GRU outputs."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
 
    def forward(self, gru_outputs):
        """
        gru_outputs: (batch_size, seq_length, hidden_dim)
        returns: context_vector (batch_size, hidden_dim), attn_weights (batch_size, seq_length)
        """
        scores = self.attention(gru_outputs).squeeze(-1)  # (batch, seq)
        attn_weights = torch.softmax(scores, dim=1)  # (batch, seq)
        context = torch.bmm(attn_weights.unsqueeze(1), gru_outputs).squeeze(1)  # (batch, hidden)
        return context, attn_weights
 
 
class TextBiGRUModel(nn.Module):
    """BiGRU with attention for emotion classification."""
    def __init__(self, vocab_size: int, embed_dim: int = 200, hidden_dim: int = 256,
                 num_classes: int = 7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # FIX: dropout 0.2 (was 0.4 — too aggressive)
        self.dropout1 = nn.Dropout(0.2)
 
        self.bigru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
 
        self.attention = Attention(hidden_dim * 2)
 
        # FIX: dropout 0.15 (was 0.4)
        self.dropout2 = nn.Dropout(0.15)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
 
    def forward(self, x):
        embedded = self.dropout1(self.embedding(x))  # (batch, seq, embed)
        gru_out, _ = self.bigru(embedded)  # (batch, seq, hidden*2)
        context, attn_weights = self.attention(gru_out)  # (batch, hidden*2)
        out = self.dropout2(context)
        logits = self.fc(out)  # (batch, num_classes)
        return logits, attn_weights
 
 
# ── Training ──────────────────────────────────────────────────────────────────
def train_bigru_pipeline() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_METRICS, exist_ok=True)
    os.makedirs(RESULTS_PLOTS, exist_ok=True)
 
    print("\n📚 Initializing BiGRU Text Pipeline & Building Vocabulary...")
 
    # Build datasets
    train_dataset = TextEmotionDataset(
        TEXT_TRAIN_CSV,
        is_training=True,
        max_length=50,
        build_vocab=True,
    )
    dev_dataset = TextEmotionDataset(
        TEXT_DEV_CSV,
        is_training=False,
        max_length=50,
        label_encoder=train_dataset.label_encoder,
        vocab=train_dataset.vocab,
        build_vocab=False,
    )
 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
    dev_loader   = DataLoader(dev_dataset,   batch_size=32, shuffle=False, num_workers=0)
 
    print(f"Train samples  : {len(train_dataset)}")
    print(f"Dev samples    : {len(dev_dataset)}")
    print(f"Vocab size     : {len(train_dataset.vocab)}")
    print(f"Classes        : {list(train_dataset.label_encoder.classes_)}")
 
    # FIX: save encoder and vocab immediately
    joblib.dump(train_dataset.label_encoder, LABEL_ENC_PATH)
    with open(TEXT_VOCAB_PATH, "w") as f:
        json.dump(train_dataset.vocab, f)
    print(f"✅ LabelEncoder saved → {LABEL_ENC_PATH}")
    print(f"✅ Vocabulary saved → {TEXT_VOCAB_PATH}")
 
    # Class weights
    df_train = pd.read_csv(TEXT_TRAIN_CSV)
    if 'Emotion' in df_train.columns:
        df_train['emotion'] = df_train['Emotion']
    df_train['emotion'] = df_train['emotion'].str.lower().replace(LABEL_MAP)
    df_train = df_train[df_train['emotion'].isin(VALID_EMOTIONS)]
 
    class_order   = train_dataset.label_encoder.classes_
    raw_weights   = compute_class_weight('balanced', classes=class_order,
                                         y=df_train['emotion'].values)
    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
    print(f"Class weights  : {dict(zip(class_order, raw_weights.round(3)))}")
 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
 
    model = TextBiGRUModel(vocab_size=len(train_dataset.vocab), num_classes=7).to(device)
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
 
    # FIX: cosine LR schedule
    num_epochs = 20
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
 
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'dev_acc': []}
    best_dev_acc  = 0.0
    best_epoch    = 0
    patience      = 3
    epochs_no_improve = 0
 
    print("\n📚 Starting BiGRU fine-tuning with early stopping...")
    for epoch in range(1, num_epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0
 
        for batch_idx, (sequences, labels) in enumerate(train_loader, 1):
            sequences = sequences.to(device)
            labels    = labels.to(device)
 
            optimizer.zero_grad()
            logits, _ = model(sequences)
            loss      = criterion(logits, labels)
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item() * sequences.size(0)
            correct      += (logits.argmax(1) == labels).sum().item()
            total        += labels.size(0)
 
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch}/{num_epochs}  Batch {batch_idx}"
                      f"  Loss: {running_loss/total:.4f}"
                      f"  Acc: {correct/total*100:.2f}%")
 
        train_loss = running_loss / total
        train_acc  = correct / total * 100
 
        # ── Dev evaluation (no test peeking) ───────────────────
        model.eval()
        dev_correct, dev_total = 0, 0
        with torch.no_grad():
            for sequences, labels in dev_loader:
                sequences = sequences.to(device)
                labels    = labels.to(device)
                preds     = model(sequences)[0].argmax(1)
                dev_correct += (preds == labels).sum().item()
                dev_total   += labels.size(0)
        dev_acc = dev_correct / dev_total * 100
 
        scheduler.step()
 
        print(f"Epoch [{epoch}/{num_epochs}]  "
              f"Train Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.2f}%  "
              f"Dev Acc: {dev_acc:.2f}%")
 
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['dev_acc'].append(dev_acc)
 
        # FIX: save best checkpoint based on dev accuracy + early stopping
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch   = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), TEXT_MODEL_PATH)
            print(f"  ✅ Best model saved (dev acc: {best_dev_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch} (no improve for {patience} epochs)")
                break
 
    print(f"\nTraining complete. Best dev acc: {best_dev_acc:.2f}% at epoch {best_epoch}")
 
    # ── Save history CSV ───────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(RESULTS_METRICS, "bigru_text_metrics_history.csv"), index=False)
 
    # ── Save plots ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    axes[0].plot(history['epoch'], history['train_loss'], marker='o', color='red',   label='Train Loss')
    axes[0].set_title('BiGRU Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
 
    axes[1].plot(history['epoch'], history['train_acc'], marker='o', color='blue',  label='Train Acc')
    axes[1].plot(history['epoch'], history['dev_acc'],   marker='s', color='green', label='Dev Acc')
    axes[1].set_title('BiGRU Training vs Dev Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS, "bigru_text_training_history.png"), dpi=150)
    plt.close()
 
    print(f"✅ Plots saved → Results/plots/bigru_text_training_history.png")
    print(f"✅ Metrics saved → Results/metrics/bigru_text_metrics_history.csv")
 
 
if __name__ == "__main__":
    train_bigru_pipeline()