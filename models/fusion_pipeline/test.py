"""
LATE FUSION MULTIMODAL MODEL - TESTING SCRIPT

CORRECT COMMAND TO RUN:
    python models/fusion_pipeline/test.py

Run from: Project Root Directory
Prerequisite: Train fusion model first using train.py
This script evaluates the fusion model and generates confusion matrix & F1 scores.
Compares audio-only, text-only, and fusion model performance.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==========================================
# PATHS & CONFIG
# ==========================================
BASE_DIR          = Path('.')
MANIFEST_PATH     = BASE_DIR / 'tess_manifest.csv'
MODELS_DIR        = BASE_DIR / 'models' / 'fusion_pipeline'
FUSION_MODEL_PATH = MODELS_DIR / 'tess_fusion_model.pth'
LABEL_ENC_PATH    = MODELS_DIR / 'fusion_label_encoder.pkl'
TEXT_MODEL_PATH   = BASE_DIR / 'models' / 'text_pipeline' / 'tess_text_model_OAF.joblib'
RESULTS_METRICS   = BASE_DIR / 'Results' / 'metrics'
RESULTS_PLOTS     = BASE_DIR / 'Results' / 'plots'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import core classes from the sibling train.py module
sys.path.insert(0, str(BASE_DIR / 'models' / 'fusion_pipeline'))
from train import (  # type: ignore
    SpeechEmotionModel, FusionHead, TESSFusionModel,
    TESSFusionDataset
)

# ==========================================
# PLOTTING HELPERS
# ==========================================
def plot_comparison(audio_acc: float, text_acc: float, fusion_acc: float, save_path: Path):
    models = ['Audio Only\n(BiGRU)', 'Text Only\n(TF-IDF+LR)', 'Fusion\n(Audio+Text)']
    accs   = [audio_acc * 100, text_acc * 100, fusion_acc * 100]
    colours = ['#2196F3', '#FF9800', '#4CAF50']

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(models, accs, color=colours, edgecolor='white', width=0.5)
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        if height >= 10:
            y = height - 2
            va = 'center'
            color = 'white'
        else:
            y = height + 1
            va = 'bottom'
            color = 'black'
        ax.text(bar.get_x() + bar.get_width() / 2, y, f'{acc:.2f}%',
                ha='center', va=va, fontsize=12, fontweight='bold', color=color)
    ax.set_title('TESS Model Comparison: Audio vs Text vs Fusion', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 110)
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion_and_f1(all_labels, all_preds, class_names, cm_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Fusion Model: Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Emotion Label')
    axes[0].tick_params(axis='x', rotation=45)

    # F1 Score Bar Chart
    f1_scores = f1_score(all_labels, all_preds, average=None)
    colors = ['#e74c3c' if f < 0.4 else '#f39c12' if f < 0.6 else '#2ecc71' for f in f1_scores]
    bars = axes[1].bar(class_names, f1_scores, color=colors, edgecolor='white')
    axes[1].set_title('Fusion Model: Per-class F1 Score', pad=18)
    axes[1].set_xlabel('Emotion')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_ylim(0, 1.12)
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].tick_params(axis='x', labelsize=10)
    
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        if height >= 0.25:
            y = height - 0.06
            va = 'center'
            color = 'white'
        else:
            y = height + 0.02
            va = 'bottom'
            color = 'black'
        axes[1].text(bar.get_x() + bar.get_width() / 2, y,
                     f'{score:.2f}', ha='center', va=va, fontsize=9, color=color)

    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.25)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(cm_path, dpi=150)
    plt.close()

def plot_tsne(embeddings, labels, class_names, save_path):
    print("Computing 2D t-SNE coordinate transformations from Fusion features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(class_names):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1], label=cls, alpha=0.7, s=25)
        
    ax.set_title('t-SNE Visualization: Fusion Model Learned Representations')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(title='Emotions', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN EVALUATION
# ==========================================
def evaluate_fusion():
    RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
    RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

    le = joblib.load(LABEL_ENC_PATH)
    class_names = list(le.classes_)

    text_bundle = joblib.load(TEXT_MODEL_PATH)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        text_bundle['sid'] = SentimentIntensityAnalyzer()
    except ImportError:
        text_bundle['sid'] = None

    df = pd.read_csv(MANIFEST_PATH)
    df['emotion'] = df['emotion'].str.lower().str.strip().replace({'ps': 'surprise'})
    _, temp_df = train_test_split(df, test_size=0.4, stratify=df['emotion'], random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['emotion'], random_state=42)
    
    print(f"Loading Test Dataset (Reserved 20%: {len(test_df)} samples)...")
    test_dataset = TESSFusionDataset(test_df, le, text_bundle)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    audio_model  = SpeechEmotionModel().to(device)
    fusion_head  = FusionHead(num_classes=7).to(device)
    fusion_model = TESSFusionModel(audio_model, fusion_head).to(device)
    fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device))
    fusion_model.eval()

    all_preds, all_labels_list, all_embeds = [], [], []
    audio_preds_only, text_preds_only = [], []

    print("Running inference on test set...")
    with torch.no_grad():
        for mfcc, text_probs, labels in test_loader:
            mfcc, text_probs = mfcc.to(device), text_probs.to(device)

            logits, embed = fusion_model(mfcc, text_probs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels_list.extend(labels.numpy())
            all_embeds.extend(embed.cpu().numpy())

            audio_logits, _ = fusion_model.audio_model(mfcc)
            audio_preds_only.extend(audio_logits.argmax(1).cpu().numpy())
            text_preds_only.extend(text_probs.argmax(1).cpu().numpy())

    embeddings = np.array(all_embeds)
    fusion_acc = accuracy_score(all_labels_list, all_preds)
    audio_acc  = accuracy_score(all_labels_list, audio_preds_only)
    text_acc   = accuracy_score(all_labels_list, text_preds_only)

    report_str = classification_report(all_labels_list, all_preds, target_names=class_names, digits=4, zero_division=0)
    
    print("\n============= TESS FUSION MODEL — TEST RESULTS =============")
    print(f"Audio-only Accuracy  : {audio_acc*100:.2f}%")
    print(f"Text-only  Accuracy  : {text_acc*100:.2f}%")
    print(f"Fusion     Accuracy  : {fusion_acc*100:.2f}%")
    print("\n" + report_str)

    with open(RESULTS_METRICS / 'tess_fusion_performance.txt', 'w', encoding='utf-8') as f:
        f.write("=== TESS FUSION MODEL — TEST RESULTS ===\n")
        f.write("Architecture: MFCCs->BiGRU + TF-IDF/LogReg -> Late Fusion\n\n")
        f.write(f"Audio-only Accuracy : {audio_acc*100:.2f}%\n")
        f.write(f"Text-only  Accuracy : {text_acc*100:.2f}%\n")
        f.write(f"Fusion     Accuracy : {fusion_acc*100:.2f}%\n\n")
        f.write(report_str)

    plot_confusion_and_f1(all_labels_list, all_preds, class_names, RESULTS_PLOTS / 'tess_fusion_test_evaluation.png')
    plot_tsne(embeddings, np.array(all_labels_list), class_names, RESULTS_PLOTS / 'tess_fusion_tsne.png')
    plot_comparison(audio_acc, text_acc, fusion_acc, RESULTS_PLOTS / 'tess_fusion_comparison.png')

    print("✅ Fusion evaluation complete! Metrics, Dual-Plot, Comparison, and t-SNE saved.")

if __name__ == "__main__":
    evaluate_fusion()