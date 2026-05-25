import os
import sys
from datetime import datetime
import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Set the BASE_DIR (Root of your project) first!
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Add that Root folder to Python's system path so it can find the 'models' module
sys.path.insert(0, BASE_DIR)

# 3. NOW we can safely import from models
from models.text_pipeline.train_bigru import TextBiGRUModel

# Set the rest of the Paths
TEXT_VOCAB_PATH = os.path.join(BASE_DIR, "models", "text_pipeline", "text_vocab.json")
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text_pipeline", "text_bigru_model.pth")
RESULTS_PLOTS_DIR = os.path.join(BASE_DIR, "Results", "plots")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def explain_model_decision(custom_sentence, output_name=None):
    # 1. Load the Vocabulary
    with open(TEXT_VOCAB_PATH, "r") as f:
        vocab = json.load(f)
        
    # 2. Load the Model
    model = TextBiGRUModel(vocab_size=len(vocab), num_classes=7).to(device)
    model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 3. Clean and Tokenize the custom sentence
    clean_text = re.sub(r'[^\w\s]', '', custom_sentence.lower())
    words = clean_text.split()
    
    if len(words) == 0:
        print("Please enter a valid sentence.")
        return
        
    word_ids = [vocab.get(w, 1) for w in words] # 1 is <UNK>
    
    # Pad to 50
    padded_ids = word_ids + [0] * (50 - len(word_ids))
    sequence_tensor = torch.tensor([padded_ids], dtype=torch.long).to(device)

    # 4. Run Inference to get Predictions AND Attention Weights
    with torch.no_grad():
        outputs, attn_weights = model(sequence_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        
        # Extract only the weights for the actual words (ignore the <PAD> zeros)
        actual_weights = attn_weights.squeeze().cpu().numpy()[:len(words)]
        
    predicted_idx = np.argmax(probabilities)
    predicted_emotion = emotions[predicted_idx]
    confidence = probabilities[predicted_idx] * 100

    print(f"\nSentence: '{custom_sentence}'")
    print(f"Prediction: {predicted_emotion.upper()} ({confidence:.1f}% confidence)")

    # 5. Generate the XAI Heatmap Plot
    plt.figure(figsize=(10, 2))
    sns.heatmap([actual_weights], annot=[words], fmt="", cmap="Reds", 
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=False, yticklabels=False)
    
    plt.title(f"Model Focus (Predicted: {predicted_emotion.capitalize()})")
    plt.tight_layout()
    
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"attention_heatmap_{timestamp}.png"
    save_path = os.path.join(RESULTS_PLOTS_DIR, output_name)
    plt.savefig(save_path)
    print(f"✅ Explainability Heatmap saved to: {save_path}")

if __name__ == "__main__":
    # It should zero in on "furious" and ignore the rest
    test_sentence = "I am so sorry to hear that terrible news."
    explain_model_decision(test_sentence)