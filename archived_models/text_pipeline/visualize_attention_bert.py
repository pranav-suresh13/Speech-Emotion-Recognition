import os
import sys
from datetime import datetime
import re
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn

# 1. Set the BASE_DIR (Root of your project) first!
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Add that Root folder to Python's system path so it can find the 'models' module
sys.path.insert(0, BASE_DIR)

# Set the rest of the Paths
BERT_MODEL_PATH = os.path.join(BASE_DIR, "archived_models", "text_pipeline", "bert_model.pth")
LABEL_ENC_PATH = os.path.join(BASE_DIR, "archived_models", "text_pipeline", "label_encoder.pkl")
RESULTS_PLOTS_DIR = os.path.join(BASE_DIR, "archived_models", "Results", "plots")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Simple BERT model for inference
class BERTEmotionClassifier(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', output_attentions=True)
        
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(cls)), out.attentions


def explain_bert_decision(custom_sentence, output_name=None):
    """Simple BERT attention visualization - similar to BigRU version"""
    # 1. Load the Tokenizer and Model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = BERTEmotionClassifier(num_classes=7).to(device)
    model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Load Label Encoder
    label_encoder = joblib.load(LABEL_ENC_PATH)

    # 3. Clean and Tokenize the sentence
    clean_text = re.sub(r'[^\w\s]', '', custom_sentence.lower())
    words = clean_text.split()
    
    if len(words) == 0:
        print("Please enter a valid sentence.")
        return
    
    # Tokenize with BERT tokenizer
    encoding = tokenizer(
        custom_sentence,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    # 4. Get predictions and attention
    with torch.no_grad():
        logits, attn_list = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Use the last layer's attention, average across heads
        last_attn = attn_list[-1][0].cpu().numpy()  # [num_heads, seq_len, seq_len]
        avg_attn = last_attn.mean(axis=0)  # Average across heads
        
    predicted_idx = np.argmax(probabilities)
    predicted_emotion = emotions[predicted_idx]
    confidence = probabilities[predicted_idx] * 100
    
    # Get actual token length (remove padding)
    actual_length = attention_mask[0].sum().item()
    actual_tokens = tokens[:actual_length]
    
    # Get attention weights for [CLS] token (first row) - shows what model attends to
    cls_attention = avg_attn[0, :actual_length]  # CLS attends to all tokens
    
    # Simplify tokens by removing ## markers and special tokens
    display_tokens = []
    display_weights = []
    for i, token in enumerate(actual_tokens):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            clean_token = token.replace('##', '')
            display_tokens.append(clean_token)
            display_weights.append(cls_attention[i])

    print(f"\nSentence: '{custom_sentence}'")
    print(f"Tokens: {display_tokens}")
    print(f"Prediction: {predicted_emotion.upper()} ({confidence:.1f}% confidence)")

    # 5. Generate Heatmap (like BigRU - simple 1D visualization)
    plt.figure(figsize=(14, 2))
    sns.heatmap([display_weights], 
                annot=[display_tokens],
                fmt="",
                cmap="Reds",
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=False,
                yticklabels=False)
    
    plt.title(f"Model Focus (Predicted: {predicted_emotion.capitalize()})")
    plt.tight_layout()
    
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"bert_attention_heatmap_{timestamp}.png"
    
    save_path = os.path.join(RESULTS_PLOTS_DIR, output_name)
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ BERT Attention Heatmap saved to: {save_path}")
    plt.close()



if __name__ == "__main__":
    # Example usage - similar to BigRU
    test_sentence = "I love this so much"
    
    print("=" * 70)
    print("BERT EMOTION MODEL - ATTENTION VISUALIZATION")
    print("=" * 70)
    
    explain_bert_decision(test_sentence)
    
    print("\n✅ Attention visualization completed!")
