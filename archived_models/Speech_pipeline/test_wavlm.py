"""
WavLM SPEECH EMOTION RECOGNITION - TESTING SCRIPT

CORRECT COMMAND TO RUN THIS SCRIPT:
    python archived_models/Speech_pipeline/test_wavlm.py

Run from: Project Root Directory
Prerequisite: Train model first using train_wavlm.py
This script evaluates the WavLM model and saves results to archived_models/Results/
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import sys

# Set paths relative to script location for portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Project root
ARCHIVED_MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # archived_models folder

MANIFEST_PATH = os.path.join(BASE_DIR, "tess_manifest.csv")
MODEL_LOAD_PATH = os.path.join(ARCHIVED_MODELS_DIR, "Speech_pipeline", "wavlm_random_split.pth")
RESULTS_METRICS_DIR = os.path.join(ARCHIVED_MODELS_DIR, "Results", "metrics")
RESULTS_PLOTS_DIR = os.path.join(ARCHIVED_MODELS_DIR, "Results", "plots")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_wavlm import WavLMDataset, WavLMClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_wavlm_pipeline():
    # Load manifest and replicate the EXACT 60-20-20 split from train.py
    df = pd.read_csv(MANIFEST_PATH)
    label_encoder = LabelEncoder()
    label_encoder.fit(df['emotion'])
    
    _, temp_df = train_test_split(df, test_size=0.40, stratify=df['emotion'], random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['emotion'], random_state=42)
    
    print(f"Loading WavLM Test Dataset (Reserved 20%: {len(test_df)} samples)...")
    test_dataset = WavLMDataset(test_df, label_encoder, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", use_safetensors=True).to(device)
    wavlm_backbone.eval()
    
    model = WavLMClassifier(num_classes=7).to(device)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    model.eval()
    
    all_preds, all_labels, all_embeddings = [], [], []
    
    print("Running WavLM inference on test set...")
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms = waveforms.to(device)
            inputs = processor(list(waveforms.cpu().numpy()), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
            features = wavlm_backbone(inputs).last_hidden_state
            
            outputs, temporal_repr = model(features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_embeddings.extend(temporal_repr.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_embeddings = np.array(all_embeddings)
    emotion_names = list(test_dataset.label_encoder.classes_)
            
    report_string = classification_report(all_labels, all_preds, target_names=emotion_names)
    print("\n================ WAVLM TEST SET PERFORMANCE ================")
    print(report_string)
    
    # === SAVE TEXT METRICS ===
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_METRICS_DIR, "wavlm_test_performance.txt"), "w") as f:
        f.write("=== WAVLM TEST SET RESULTS (Random Split) ===\n\n")
        f.write(report_string)
        
    # === DUAL PLOT: CONFUSION MATRIX & F1 SCORE ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=emotion_names, yticklabels=emotion_names, ax=axes[0])
    axes[0].set_title('WavLM Model — Confusion Matrix')
    axes[0].set_ylabel('True Emotion Label')
    axes[0].set_xlabel('Predicted Emotion Label')
    axes[0].tick_params(axis='x', rotation=45)

    # Right: F1 Score Bar Chart
    f1_scores = f1_score(all_labels, all_preds, average=None)
    colors = ['#e74c3c' if s < 0.35 else '#f39c12' if s < 0.55 else '#2ecc71' for s in f1_scores]
    
    bars = axes[1].bar(emotion_names, f1_scores, color=colors)
    axes[1].set_title('WavLM Model — Per-class F1 Score')
    axes[1].set_xlabel('Emotion')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars, f1_scores):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{score:.2f}', ha='center', va='bottom', fontsize=10)

    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, "wavlm_test_evaluation.png"), dpi=150)
    plt.close()
    
    # === T-SNE CLUSTERING PLOT ===
    print("Computing 2D t-SNE coordinate transformations from WavLM features...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(10, 8))
    for i, emotion in enumerate(emotion_names):
        indices = np.where(all_labels == i)[0]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=emotion, alpha=0.7)
    
    plt.title('t-SNE Visualization: WavLM Feature Representations')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title="Emotions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, "wavlm_tsne_temporal.png"), dpi=150)
    plt.close()

    print("✅ Testing complete! Metrics, Dual-Plot, and t-SNE saved.")

if __name__ == "__main__":
    evaluate_wavlm_pipeline()