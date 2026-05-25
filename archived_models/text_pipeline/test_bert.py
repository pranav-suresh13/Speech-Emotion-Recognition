import os
import sys
import joblib

import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import DistilBertTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVED_MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
if not os.path.exists(os.path.join(BASE_DIR, "tess_manifest.csv")):
    print("\n❌ ERROR: Run from project root:  python archived_models/text_pipeline/test_bert.py")
    sys.exit(1)
 
TEXT_TEST_CSV   = os.path.join(BASE_DIR, "data", "MELD", "test_sent_emo.csv")
MODELS_DIR      = os.path.join(ARCHIVED_MODELS_DIR, "text_pipeline")
BERT_MODEL_PATH = os.path.join(MODELS_DIR, "bert_model.pth")
LABEL_ENC_PATH  = os.path.join(MODELS_DIR, "label_encoder.pkl")   # ← saved by train_bert.py
RESULTS_METRICS = os.path.join(ARCHIVED_MODELS_DIR, "Results", "metrics")
RESULTS_PLOTS   = os.path.join(ARCHIVED_MODELS_DIR, "Results", "plots")
# Dynamically import the dataset and model architecture from train_bert
from train_bert import BERTEmotionDataset, BERTEmotionClassifier # type: ignore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def evaluate_bert_pipeline() -> None:
    os.makedirs(RESULTS_METRICS, exist_ok=True)
    os.makedirs(RESULTS_PLOTS,   exist_ok=True)

    # ── Guard: encoder and model must exist ────────────────────
    if not os.path.exists(LABEL_ENC_PATH):
        print(f"\n❌ label_encoder.pkl not found at {LABEL_ENC_PATH}")
        print("   Run train_bert.py first to generate it.")
        sys.exit(1)

    if not os.path.exists(BERT_MODEL_PATH):
        print(f"\n❌ bert_model.pth not found at {BERT_MODEL_PATH}")
        print("   Run train_bert.py first.")
        sys.exit(1)

    # ── Load resources ─────────────────────────────────────────
    label_encoder = joblib.load(LABEL_ENC_PATH)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    print(f"✅ LabelEncoder loaded — classes: {list(label_encoder.classes_)}")
    print(f"✅ DistilBERT Tokenizer loaded")

    # ── Prepare Dataset ────────────────────────────────────────
    test_dataset = BERTEmotionDataset(
        TEXT_TEST_CSV,
        tokenizer=tokenizer,
        max_length=128,
        label_encoder=label_encoder
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    # ── Load model ─────────────────────────────────────────────
    model = BERTEmotionClassifier(num_classes=7).to(device)
    model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
    model.eval()
    print("✅ BERT Model loaded")

    # ── Inference ──────────────────────────────────────────────
    all_preds, all_labels = [], []

    print("\nRunning inference on MELD test set...")
    with torch.no_grad():
        for input_ids, attn_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            
            logits = model(input_ids, attn_mask)
            preds  = logits.argmax(dim=1)
            
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

    print("\n================ BERT TEXT-ONLY PERFORMANCE ================")
    print(f"Overall Accuracy : {overall_acc*100:.2f}%")
    print(report_str)

    # ── Save metrics ───────────────────────────────────────────
    metrics_path = os.path.join(RESULTS_METRICS, "bert_text_only_performance.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== DistilBERT TEXT-ONLY MODEL RESULTS ===\n")
        f.write("Dataset: MELD (Text)\n")
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
    axes[0].set_title('BERT Text — Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].tick_params(axis='x', rotation=45)

    # Per-class F1 bar chart
    colours = ['#e74c3c' if f < 0.4 else '#f39c12' if f < 0.6 else '#27ae60'
               for f in per_class_f1]
    axes[1].bar(class_names, per_class_f1, color=colours, edgecolor='white')
    axes[1].set_title('BERT Text — Per-class F1 Score')
    axes[1].set_xlabel('Emotion')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(per_class_f1):
        axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_PLOTS, "bert_text_confusion_matrix.png")
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
    print("\n✅ BERT Evaluation complete!")


if __name__ == "__main__":
    evaluate_bert_pipeline()