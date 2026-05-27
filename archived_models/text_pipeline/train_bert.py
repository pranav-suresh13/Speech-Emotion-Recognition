import os
import sys
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
from transformers import DistilBertTokenizer, DistilBertModel
 
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVED_MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
if not os.path.exists(os.path.join(BASE_DIR, "tess_manifest.csv")):
    print("\n❌ ERROR: Run from project root:  python archived_models/text_pipeline/train_bert.py")
    sys.exit(1)
 
TEXT_TRAIN_CSV   = os.path.join(BASE_DIR, "data", "MELD", "train_sent_emo_balanced.csv")
TEXT_DEV_CSV     = os.path.join(BASE_DIR, "data", "MELD", "dev_sent_emo.csv")
MODELS_DIR       = os.path.join(ARCHIVED_MODELS_DIR, "text_pipeline")
BERT_MODEL_PATH  = os.path.join(MODELS_DIR, "bert_model.pth")
LABEL_ENC_PATH   = os.path.join(MODELS_DIR, "label_encoder.pkl")
RESULTS_METRICS  = os.path.join(ARCHIVED_MODELS_DIR, "Results", "metrics")
RESULTS_PLOTS    = os.path.join(ARCHIVED_MODELS_DIR, "Results", "plots")
 
VALID_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
LABEL_MAP      = {"anger": "angry", "joy": "happy", "sadness": "sad"}
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 
 
# ── Dataset ───────────────────────────────────────────────────────────────────
class BERTEmotionDataset(Dataset):
    """
    Parameters
    ----------
    label_encoder : LabelEncoder or None
        Pass a pre-fit encoder (from training) for dev/test splits.
        Pass None to fit a new encoder (training only).
    """
    def __init__(self, csv_file: str, tokenizer, max_length: int = 128,
                 label_encoder: LabelEncoder | None = None):
        df = pd.read_csv(csv_file)
 
        # Normalise column names
        if 'Utterance' in df.columns and 'text' not in df.columns:
            df['text'] = df['Utterance']
        if 'Emotion' in df.columns and 'emotion' not in df.columns:
            df['emotion'] = df['Emotion']
 
        df['emotion'] = df['emotion'].str.lower().replace(LABEL_MAP)
        df = df[df['emotion'].isin(VALID_EMOTIONS)].reset_index(drop=True)
 
        self.texts = df['text'].astype(str).values
        self.tokenizer = tokenizer
        self.max_length = max_length
 
        # Use a shared encoder — never refit on test/dev
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(VALID_EMOTIONS)   # fit on fixed class list
        else:
            self.label_encoder = label_encoder
 
        self.labels = self.label_encoder.transform(df['emotion'].values)
 
    def __len__(self) -> int:
        return len(self.texts)
 
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return (
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
 
 
# ── Model ─────────────────────────────────────────────────────────────────────
class BERTEmotionClassifier(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', output_attentions=True)

        # Freeze everything first, then unfreeze last 2 transformer blocks
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Dropout 0.2
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, return_attention=False):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [CLS] token
        logits = self.fc(self.dropout(cls))
        
        if return_attention:
            # out.attentions contains attention weights from all 6 layers
            # Average across all layers and heads for visualization
            attentions = torch.stack(out.attentions)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
            avg_attention = attentions.mean(dim=(0, 2))  # Average over layers and heads: [batch_size, seq_len, seq_len]
            return logits, avg_attention
        
        return logits


# ── Training ──────────────────────────────────────────────────────────────────
def train_bert_pipeline() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_METRICS, exist_ok=True)
    os.makedirs(RESULTS_PLOTS, exist_ok=True)
 
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
 
    # Build datasets
    train_dataset = BERTEmotionDataset(TEXT_TRAIN_CSV, tokenizer)
    dev_dataset   = BERTEmotionDataset(TEXT_DEV_CSV,   tokenizer,
                                       label_encoder=train_dataset.label_encoder)
 
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0)
    dev_loader   = DataLoader(dev_dataset,   batch_size=32, shuffle=False, num_workers=0)
 
    print(f"Train samples : {len(train_dataset)}")
    print(f"Dev samples   : {len(dev_dataset)}")
    print(f"Classes       : {list(train_dataset.label_encoder.classes_)}")
 
    # Save encoder immediately so test script can load it
    joblib.dump(train_dataset.label_encoder, LABEL_ENC_PATH)
    print(f"✅ LabelEncoder saved → {LABEL_ENC_PATH}")
 
    # Class weights computed on filtered emotions only
    df_train = pd.read_csv(TEXT_TRAIN_CSV)
    if 'Emotion' in df_train.columns:
        df_train['emotion'] = df_train['Emotion']
    df_train['emotion'] = df_train['emotion'].str.lower().replace(LABEL_MAP)
    df_train = df_train[df_train['emotion'].isin(VALID_EMOTIONS)]
 
    class_order   = train_dataset.label_encoder.classes_
    raw_weights   = compute_class_weight('balanced', classes=class_order,
                                         y=df_train['emotion'].values)
    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
    print(f"Class weights : {dict(zip(class_order, raw_weights.round(3)))}")
 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
 
    model = BERTEmotionClassifier(num_classes=7).to(device)
 
    # Two-group optimizer
    bert_params = list(model.bert.transformer.layer[-2:].parameters())
    head_params = list(model.fc.parameters())
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': 2e-5},
        {'params': head_params, 'lr': 2e-4},
    ], weight_decay=0.01)
 
    # Cosine LR schedule + increased max epochs to 15 to allow early stopping to work
    num_epochs = 15
    scheduler  = CosineAnnealingLR(optimizer, T_max=num_epochs)
 
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'dev_acc': []}
    best_dev_acc  = 0.0
    best_epoch    = 0
    
    # ── Early Stopping Variables ──
    patience = 3
    epochs_no_improve = 0
 
    print("\n📚 Starting fine-tuning with Early Stopping...")
    for epoch in range(1, num_epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0
 
        for batch_idx, (input_ids, attn_mask, labels) in enumerate(train_loader, 1):
            input_ids  = input_ids.to(device)
            attn_mask  = attn_mask.to(device)
            labels     = labels.to(device)
 
            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item() * input_ids.size(0)
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
            for input_ids, attn_mask, labels in dev_loader:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels    = labels.to(device)
                preds     = model(input_ids, attn_mask).argmax(1)
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
 
        # ── Early Stopping Logic ─────────────────────────────────
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch   = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), BERT_MODEL_PATH)
            print(f"  ✅ Best model saved (dev acc: {best_dev_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️ No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}!")
                break
 
    print(f"\nTraining complete. Best dev acc: {best_dev_acc:.2f}% at epoch {best_epoch}")
 
    # ── Save history CSV ───────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(RESULTS_METRICS, "bert_text_metrics_history.csv"), index=False)
 
    # ── Save plots ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    axes[0].plot(history['epoch'], history['train_loss'], marker='o', color='red',   label='Train Loss')
    axes[0].set_title('BERT Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
 
    axes[1].plot(history['epoch'], history['train_acc'], marker='o', color='blue',  label='Train Acc')
    axes[1].plot(history['epoch'], history['dev_acc'],   marker='s', color='green', label='Dev Acc')
    axes[1].set_title('BERT Training vs Dev Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
 
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS, "bert_text_training_history.png"))
    plt.close()
 
    print(f"✅ Plots saved → Results/plots/bert_text_training_history.png")
    print(f"✅ Metrics saved → Results/metrics/bert_text_metrics_history.csv")
 
 
if __name__ == "__main__":
    train_bert_pipeline()