"""
SPEECH EMOTION RECOGNITION - TESTING SCRIPT (MFCC + BiGRU)

CORRECT COMMAND TO RUN:
    python models/speech_pipeline/test.py

Run from: Project Root Directory
Prerequisite: Train model first using train.py
This script evaluates the MFCC + BiGRU model and saves confusion matrix & F1 scores.
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

# ==========================================
# 1. MATCHING AUDIO DATASET PREPROCESSING
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
# 2. MATCHING SPEECH MODEL ARCHITECTURE
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
# 3. EVALUATION AND VISUALIZATION ROUTINE
# ==========================================
def evaluate_speech_pipeline(manifest_path, model_path):
    if not os.path.exists(manifest_path):
        print(f"❌ ERROR: Cannot find '{manifest_path}'. Run this from the project root folder!")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model weights not found at '{model_path}'")
        return

    os.makedirs('Results/plots', exist_ok=True)
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    
    print(f"✓ Manifest loaded: {len(df)} samples")
    print(f"✓ Unique emotions: {df['emotion'].nunique()}\n")
    
    # ========== REPLICATE EXACT SPLIT FROM train.py (60-20-20) ==========
    # FIRST SPLIT: 60% train, 40% temporary
    _, temp_df = train_test_split(df, test_size=0.4, stratify=df['emotion'], random_state=42)
    
    # SECOND SPLIT: Split temp 50-50 to extract 20% test (and discard 20% val)
    _, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['emotion'], random_state=42)
    
    # Create dataset using train.py's signature (no labels parameter)
    print("Loading test dataset...")
    test_dataset = TESSSpeechDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get emotion names and num_classes from the dataset's internal encoder
    emotion_names = list(test_dataset.label_encoder.classes_)
    num_classes = len(emotion_names)
    
    # Initialize model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeechEmotionModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_embeddings = []
    
    print(f"Evaluating on test partition ({len(test_dataset)} samples) using {device}...\n")
    with torch.no_grad():
        for mfccs, labels in test_loader:
            mfccs = mfccs.to(device)
            logits, temporal_repr = model(mfccs)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_embeddings.extend(temporal_repr.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_embeddings = np.array(all_embeddings)
    
    # ------------------------------------------
    # DELIVERABLE 1: PRINT ACCURACY TEXT REPORT
    # ------------------------------------------
    # Generate the classification report as a string
    report_string = classification_report(all_labels, all_preds, target_names=emotion_names)
    
    # Print it to terminal like normal
    print("================ TEST SET PERFORMANCE ================")
    print(report_string)
    print("======================================================\n")
    
    # Auto-save the metrics to a file
    os.makedirs("Results/metrics", exist_ok=True)
    metrics_txt_path = "Results/metrics/speech_only_performance.txt"
    
    with open(metrics_txt_path, "w") as f:
        f.write("=== CUSTOM MFCC + BiGRU SPEECH MODEL RESULTS ===\n")
        f.write("Dataset: TESS Toronto Emotional Speech Set (Mixed Split)\n\n")
        f.write(report_string)
    
    print(f"✅ Successfully saved your speech report file to: {metrics_txt_path}\n")
    
    # ------------------------------------------
    # DELIVERABLE 2: DUAL PLOT (CONFUSION MATRIX & F1 SCORE)
    # ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: Confusion Matrix (Greens cmap) ---
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=emotion_names, yticklabels=emotion_names, ax=axes[0])
    axes[0].set_title('Speech-Only Model: Confusion Matrix')
    axes[0].set_ylabel('True Emotion Label')
    axes[0].set_xlabel('Predicted Emotion Label')
    axes[0].tick_params(axis='x', rotation=45)

    # --- Right: F1 Score Bar Chart ---
    f1_scores = f1_score(all_labels, all_preds, average=None)
    
    # Color coding based on score (Red < 0.35, Orange < 0.55, Green >= 0.55)
    colors = ['#e74c3c' if s < 0.35 else '#f39c12' if s < 0.55 else '#2ecc71' for s in f1_scores]
    
    bars = axes[1].bar(emotion_names, f1_scores, color=colors)
    axes[1].set_title('Speech-Only Model: Per-class F1 Score')
    axes[1].set_xlabel('Emotion')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add floating text values on top of bars
    for bar, score in zip(bars, f1_scores):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{score:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    eval_plot_path = 'Results/plots/speech_test_evaluation.png'
    plt.savefig(eval_plot_path, dpi=150)
    plt.close()
    print(f"✅ Saved dual-plot test evaluation to: {eval_plot_path}")
    
    # ------------------------------------------
    # DELIVERABLE 3: t-SNE CLUSTER GRAPH
    # ------------------------------------------
    print("Computing 2D t-SNE coordinate transformations from Temporal Block...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(10, 8))
    for i, emotion in enumerate(emotion_names):
        indices = np.where(all_labels == i)[0]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=emotion, alpha=0.7)
    
    plt.title('t-SNE Visualization: Speech Temporal Block Learned Representations')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title="Emotions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    tsne_plot_path = 'Results/plots/speech_tsne_temporal.png'
    plt.savefig(tsne_plot_path)
    plt.close()
    print(f"✅ Saved temporal block cluster visualization to: {tsne_plot_path}")

if __name__ == "__main__":
    evaluate_speech_pipeline(
      manifest_path='tess_manifest.csv',
        model_path='models/speech_pipeline/speech_only_model.pth',
    )