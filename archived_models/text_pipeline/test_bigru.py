import os
import sys
import json
import joblib
 
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVED_MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
if not os.path.exists(os.path.join(BASE_DIR, "tess_manifest.csv")):
    print("\n❌ ERROR: Run from project root:  python archived_models/text_pipeline/test_bigru.py")
    sys.exit(1)
 
TEXT_TEST_CSV   = os.path.join(BASE_DIR, "data", "MELD", "test_sent_emo.csv")
MODELS_DIR      = os.path.join(ARCHIVED_MODELS_DIR, "text_pipeline")
TEXT_VOCAB_PATH = os.path.join(MODELS_DIR, "text_vocab.json")
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_bigru_model.pth")
LABEL_ENC_PATH  = os.path.join(MODELS_DIR, "label_encoder_bigru.pkl")   # ← saved by train_bigru.py
RESULTS_METRICS = os.path.join(ARCHIVED_MODELS_DIR, "Results", "metrics")
RESULTS_PLOTS   = os.path.join(ARCHIVED_MODELS_DIR, "Results", "plots")
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bigru import TextEmotionDataset, TextBiGRUModel
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 
 
def evaluate_bigru_pipeline() -> None:
    os.makedirs(RESULTS_METRICS, exist_ok=True)
    os.makedirs(RESULTS_PLOTS,   exist_ok=True)
 
    # ── Guard: encoder and vocab must exist ────────────────────
    if not os.path.exists(LABEL_ENC_PATH):
        print(f"\n❌ label_encoder_bigru.pkl not found at {LABEL_ENC_PATH}")
        print("   Run train_bigru.py first to generate it.")
        sys.exit(1)
 
    if not os.path.exists(TEXT_VOCAB_PATH):
        print(f"\n❌ text_vocab.json not found at {TEXT_VOCAB_PATH}")
        print("   Run train_bigru.py first to generate it.")
        sys.exit(1)
 
    if not os.path.exists(TEXT_MODEL_PATH):
        print(f"\n❌ text_bigru_model.pth not found at {TEXT_MODEL_PATH}")
        print("   Run train_bigru.py first.")
        sys.exit(1)
 
    # FIX: load the encoder and vocab that were fit during training — never refit
    label_encoder = joblib.load(LABEL_ENC_PATH)
    with open(TEXT_VOCAB_PATH, "r") as f:
        vocab = json.load(f)
 
    print(f"✅ LabelEncoder loaded — classes: {list(label_encoder.classes_)}")
    print(f"✅ Vocabulary loaded — {len(vocab)} tokens")
 
    # FIX: pass pre-fit encoder so class indices are identical to training
    test_dataset = TextEmotionDataset(
        TEXT_TEST_CSV,
        is_training=False,
        max_length=50,
        label_encoder=label_encoder,
        vocab=vocab,
        build_vocab=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")
 
    # ── Load model ─────────────────────────────────────────────
    model = TextBiGRUModel(vocab_size=len(vocab), num_classes=7).to(device)
    model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded")
 
    # ── Inference ──────────────────────────────────────────────
    all_preds, all_labels = [], []
 
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            preds     = model(sequences)[0].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
 
    # ── Metrics ────────────────────────────────────────────────
    class_names    = list(label_encoder.classes_)
    overall_acc    = accuracy_score(all_labels, all_preds)
    report_str     = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
 
    print("\n================ BiGRU TEXT-ONLY PERFORMANCE ================")
    print(f"Overall Accuracy : {overall_acc*100:.2f}%")
    print(report_str)
 
    # ── Save metrics ───────────────────────────────────────────
    metrics_path = os.path.join(RESULTS_METRICS, "bigru_text_only_performance.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== BiGRU TEXT-ONLY MODEL RESULTS ===\n")
        f.write("Architecture : Embedding(200) -> BiGRU(256) -> Attention -> Dropout(0.15) -> Linear(512,7)\n")
        f.write(f"Overall Accuracy : {overall_acc*100:.2f}%\n\n")
        f.write(report_str)
    print(f"\n✅ Metrics saved → {metrics_path}")
 
    # ── Per-class accuracy bar chart + confusion matrix ────────
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = [report_dict[c]['f1-score'] for c in class_names]
 
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
 
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title('BiGRU Text — Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].tick_params(axis='x', rotation=45)
 
    # Per-class F1 bar chart
    colours = ['#e74c3c' if f < 0.4 else '#f39c12' if f < 0.6 else '#27ae60'
               for f in per_class_f1]
    axes[1].bar(class_names, per_class_f1, color=colours, edgecolor='white')
    axes[1].set_title('BiGRU Text — Per-class F1 Score')
    axes[1].set_xlabel('Emotion')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(per_class_f1):
        axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
 
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_PLOTS, "bigru_text_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✅ Plots saved → {plot_path}")
 
    # ── Print actionable summary ───────────────────────────────
    print("\n── Per-class F1 summary ──────────────────────────────────")
    for cls in class_names:
        f1  = report_dict[cls]['f1-score']
        sup = report_dict[cls]['support']
        bar = '█' * int(f1 * 20)
        flag = '⚠️ ' if f1 < 0.4 else '  '
        print(f"  {flag}{cls:<12} F1={f1:.3f}  {bar}  (n={sup})")
    print(f"\n  Weighted F1  : {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"  Macro F1     : {report_dict['macro avg']['f1-score']:.4f}")
    print(f"  Accuracy     : {overall_acc*100:.2f}%")
    print("─────────────────────────────────────────────────────────")
    print("\n✅ Evaluation complete!")
 
 
if __name__ == "__main__":
    evaluate_bigru_pipeline()