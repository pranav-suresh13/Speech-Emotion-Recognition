"""
SPEECH EMOTION RECOGNITION - TRAINING SCRIPT (MFCC + BiGRU)

CORRECT COMMAND TO RUN:
    python models/speech_pipeline/train.py

Run from: Project Root Directory
This script trains the MFCC + BiGRU model on TESS dataset.
Results and model weights are saved to Results/ and models/speech_pipeline/
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# SAFETY CHECK: Ensure script runs from project root
# ==========================================
if not os.path.exists('tess_manifest.csv'):
    print("❌ ERROR: Please run this script from the project root directory!")
    print("Example: python models/speech_pipeline/train.py")
    sys.exit(1)

# ==========================================
# DATA SPLIT STRATEGY (60-20-20)
# ==========================================
# 60% → Training set (model learns from this)
# 20% → Validation set (used during training to tune hyperparameters)
# 20% → Test set (RESERVED for test.py, never touched during training)
# This ensures proper generalization evaluation on completely unseen data.

# ==========================================
# 1. CUSTOM AUDIO DATASET WITH PREPROCESSING
# ==========================================
class TESSSpeechDataset(Dataset):
    def __init__(self, manifest_df, max_len_sec=3.0, sr=22050, n_mfcc=13):
        self.df = manifest_df
        self.max_len_sec = max_len_sec
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_pad_len = int(max_len_sec * sr)
        
        # Encode string labels to integers (0 to 6)
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['emotion'].values)
        
        # Validation: Verify labels are in valid range
        unique_labels = np.unique(self.labels)
        print(f"  Dataset: {len(self.df)} samples | Classes: {len(unique_labels)} | Label range: [{self.labels.min()}, {self.labels.max()}]")
        print(f"  Class mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        if self.labels.min() < 0 or self.labels.max() >= len(self.label_encoder.classes_):
            raise ValueError(f"Invalid labels detected! Range: [{self.labels.min()}, {self.labels.max()}], but only {len(self.label_encoder.classes_)} classes exist.")
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        file_path = self.df.iloc[idx]['path']
        label = self.labels[idx]
        
        # Preprocessing: Load and Trim Silence
        y, sr = librosa.load(file_path, sr=self.sr)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Preprocessing: Ensure uniform length (Pad or Truncate) 
        if len(y) < self.max_pad_len:
            pad_width = self.max_pad_len - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        else:
            y = y[:self.max_pad_len]
            
        # Feature Extraction: MFCCs
        # Resulting shape: (n_mfcc, time_steps)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==========================================
# 2. LIGHTWEIGHT REVOLVING SPEECH MODEL
# ==========================================
class SpeechEmotionModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_classes=7):
        super(SpeechEmotionModel, self).__init__()
        # Temporal Modelling: Bidirectional GRU to capture emotional patterns over time 
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                          num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        
        # Anti-Overfitting Strategy: Strong Dropout Layer
        self.dropout = nn.Dropout(0.4)
        
        # Classifier: Maps features to final emotion labels
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # Transpose input from (batch, features, time) to (batch, time, features) for the recurrent unit
        x = x.transpose(1, 2)
        
        # gru_out shape: (batch, time, hidden_size * 2)
        gru_out, _ = self.gru(x)
        
        # Extract temporal representation: Take the mean across the time dimension
        temporal_representation = torch.mean(gru_out, dim=1)
        
        # Apply dropout and pass to final linear classifier
        x = self.dropout(temporal_representation)
        logits = self.classifier(x)
        
        return logits, temporal_representation

# ==========================================
# 3. TRAINING ROUTINE WITH METRIC LOGGING
# ==========================================
def train_model(manifest_path):
    # Ensure Results directory exists
    os.makedirs('Results/metrics', exist_ok=True)
    os.makedirs('Results/plots', exist_ok=True)
    
    # Load manifest and execute stratified split
    df = pd.read_csv(manifest_path)
    
    # Debug: Print dataset info
    print(f"✓ Manifest loaded: {len(df)} samples")
    print(f"✓ Unique emotions: {df['emotion'].nunique()}")
    print(f"✓ Emotion distribution:\n{df['emotion'].value_counts()}\n")
    
    # ========== FIRST SPLIT: 60% train, 40% temporary ==========
    train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['emotion'], random_state=42)
    
    # ========== SECOND SPLIT: Split temp 50-50 into val (20%) and test (20%) ==========
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['emotion'], random_state=42)
    
    print(f"✓ Data Split Breakdown:")
    print(f"  - Training set:   {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  - Test set:       {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%) [RESERVED for test.py]\n")
    
    train_dataset = TESSSpeechDataset(train_df)
    val_dataset = TESSSpeechDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeechEmotionModel().to(device)
    
    # Weight decay (L2 Regularization) added to optimizer to combat memorization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    epochs = 20
    
    print(f"Beginning training loop on device: {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for mfccs, labels in train_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(mfccs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * mfccs.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += torch.sum(preds == labels).item()
            
        # Validation Pass
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for mfccs, labels in val_loader:
                mfccs, labels = mfccs.to(device), labels.to(device)
                logits, _ = model(mfccs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * mfccs.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += torch.sum(preds == labels).item()
                
        # Calculate summary epoch values
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = train_correct / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_correct / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        # Append data to fulfill report obligations 
        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        })
        
    # Save training logs to CSV for direct graphing in your report
    history_df = pd.DataFrame(history)
    history_df.to_csv('Results/metrics/speech_metrics_history.csv', index=False)
    
    # Save the final model parameters safely inside the designated folder
    os.makedirs('models/speech_pipeline', exist_ok=True)
    torch.save(model.state_dict(), 'models/speech_pipeline/speech_only_model.pth')
    
    # ==========================================
    # 4. PLOT GENERATION (BiGRU Style)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left Panel: Loss
    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', color='red', marker='o')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', color='orange', marker='x')
    axes[0].set_title('Speech BiGRU Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Right Panel: Accuracy (Multiplied by 100 for percentage scale)
    axes[1].plot(history_df['epoch'], history_df['train_acc'] * 100, label='Train Acc', color='blue', marker='o')
    axes[1].plot(history_df['epoch'], history_df['val_acc'] * 100, label='Val Acc', color='green', marker='s')
    axes[1].set_title('Speech BiGRU Training vs Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('Results/plots/speech_training_history.png', dpi=150)
    plt.close()

    print("\n✅ Training phase finished. Model, Metrics CSV, and Dual-Plots saved successfully.")

if __name__ == "__main__":
    # Ensure you pass your generated data manifest csv file path here
    train_model('tess_manifest.csv')