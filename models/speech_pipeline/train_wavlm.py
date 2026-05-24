import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set paths relative to script location for portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_PATH = os.path.join(BASE_DIR, "tess_manifest.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "speech_pipeline", "wavlm_random_split.pth")
RESULTS_METRICS_DIR = os.path.join(BASE_DIR, "Results", "metrics")
RESULTS_PLOTS_DIR = os.path.join(BASE_DIR, "Results", "plots")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for WavLM training: {device}")

# Load WavLM Backbone and freeze it
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
wavlm_backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", use_safetensors=True).to(device)

for param in wavlm_backbone.parameters():
    param.requires_grad = False

class WavLMDataset(Dataset):
    def __init__(self, df, label_encoder, target_sr=16000, max_duration=3.0, augment=False):
        self.df = df.reset_index(drop=True)
        self.target_sr = target_sr
        self.max_samples = int(target_sr * max_duration)
        self.augment = augment
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, sr = librosa.load(row['path'], sr=self.target_sr)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) < self.max_samples:
            y = np.pad(y, (0, self.max_samples - len(y)), 'constant')
        else:
            y = y[:self.max_samples]
            
        y = (y - np.mean(y)) / (np.std(y) + 1e-6) # Normalization
        
        if self.augment and np.random.rand() < 0.5: # Augmentation
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            y = y + noise_amp * np.random.normal(size=y.shape[0])
            
        label = self.label_encoder.transform([row['emotion']])[0]
        return torch.tensor(y, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class WavLMClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(WavLMClassifier, self).__init__()
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        temporal_representation = self.dropout2(x)
        # Return both logits and temporal representation for t-SNE
        logits = self.out(temporal_representation)
        return logits, temporal_representation

def train_wavlm_pipeline():
    # Load manifest and encode labels
    df = pd.read_csv(MANIFEST_PATH)
    label_encoder = LabelEncoder()
    label_encoder.fit(df['emotion'])
    
    # === 60-20-20 STRATIFIED SPLIT ===
    train_df, temp_df = train_test_split(df, test_size=0.40, stratify=df['emotion'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['emotion'], random_state=42)
    
    print(f"Data Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (Reserved)")

    train_dataset = WavLMDataset(train_df, label_encoder, augment=True)
    val_dataset = WavLMDataset(val_df, label_encoder, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = WavLMClassifier(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    best_val_acc = 0.0
    wavlm_backbone.eval()
    
    num_epochs = 10
    print("\nBeginning WavLM Random Split Training...")
    for epoch in range(num_epochs):
        # -- TRAIN PASS --
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            with torch.no_grad():
                inputs = processor(list(waveforms.cpu().numpy()), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
                features = wavlm_backbone(inputs).last_hidden_state
                
            optimizer.zero_grad()
            predictions, _ = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * waveforms.size(0)
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = (correct / total) * 100
        
        # -- VAL PASS --
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)
                inputs = processor(list(waveforms.cpu().numpy()), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
                features = wavlm_backbone(inputs).last_hidden_state
                
                predictions, _ = model(features)
                loss = criterion(predictions, labels)
                
                val_loss += loss.item() * waveforms.size(0)
                _, predicted = torch.max(predictions, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    # === AUTO-SAVE ASSETS ===
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(RESULTS_METRICS_DIR, "wavlm_metrics_history.csv"), index=False)
    
    # === PLOTS ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="red", marker="o")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="orange", marker="x")
    axes[0].set_title("WavLM Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(history["epoch"], history["train_acc"], label="Train Acc", color="blue", marker="o")
    axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc", color="green", marker="s")
    axes[1].set_title("WavLM Training vs Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, "wavlm_training_history.png"))
    print("✅ Training complete! Model, Metrics CSV, and Dual-Plots saved.")

if __name__ == "__main__":
    train_wavlm_pipeline()