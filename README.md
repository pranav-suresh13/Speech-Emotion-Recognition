# Speech Emotion Recognition

This project is a multimodal emotion recognition system that combines speech and text signals through separate and fusion pipelines. It includes individual models for speech, text, and a late-fusion system that integrates both modalities, along with comprehensive metrics and visualizations for evaluation.

## Quick Overview
If you are reviewing this project for the first time, the repository is structured around key directories and files.

**Important Note**: Due to GitHub's file size limits, all heavy assets have been zipped and hosted externally. Please download the required files from the link below (see Setup section for details):
https://drive.google.com/drive/folders/16SWmrJ-QVp3ogzqmuLv9SByIkySN3pQL?usp=sharing

- `data/`: Raw datasets (MELD and TESS) used for training and testing.
- `models/`: Training scripts, evaluation scripts, and the pre-trained weights/supporting artifacts for each pipeline.
- `archived_models/`: Development and alternative model variants (previous experiments and iterations).
- `Results/`: Evaluation outputs such as metrics tables, confusion matrices, and visualizations. 
- `requirements.txt`: Python dependencies required to run the code locally.

## Datasets

### MELD (Multimodal EmotionLines Dataset)

| Property | Details |
| :--- | :--- |
| **Modality** | Text-based (transcripts from TV show Friends) |
| **Emotions** | 7 classes: Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise |
| **Files** | `train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv` |
| **Splits** | Train (7,215), Dev (2,406), Test (4,040) samples |
| **Preprocessed** | `train_sent_emo_balanced.csv` (augmented training set) |
| **Use Case** | Training alternative text models (DistilBERT, BiGRU) |
| **Location** | `data/MELD/` |

### TESS (Toronto Emotional Speech Set)

| Property | Details |
| :--- | :--- |
| **Modality** | Audio (WAV files) with transcripts |
| **Emotions** | 7 emotions: Anger, Disgust, Fear, Happiness, Neutral, Pleasant Surprise, Sadness |
| **Speakers** | 2 actors (OAF - Older Adult Female, YAF - Young Adult Female) |
| **Total Samples** | 2,800 audio files (200 files per emotion per speaker) |
| **Preprocessed** | Auto-generated `tess_manifest.csv` (maps files to emotion labels) |
| **Use Cases** | Training all production models (MFCC+BiGRU, TF-IDF+LR, Late Fusion) |
| **Location** | `data/TESS Toronto emotional speech set data/` |

## Emotion Classes

This project recognizes **7 emotion categories** across all datasets:

| Emotion | Description |
|---------|-------------|
| **Anger** | Expression of intense displeasure or hostility |
| **Disgust** | Expression of disapproval or revulsion |
| **Fear** | Expression of apprehension or anxiety |
| **Happiness** | Expression of joy or contentment |
| **Neutral** | Absence of strong emotion |
| **Sadness** | Expression of sorrow or melancholy |
| **Surprise** | Expression of astonishment (pleasant or unpleasant) |

## Pipelines & Models Overview

### Speech Pipeline
Acoustic emotion recognition models trained on the TESS (Toronto Emotional Speech Set) dataset.

- **MFCC + BiGRU**: A Bidirectional GRU trained on MFCC (Mel-Frequency Cepstral Coefficients) features. 
  - **Model Type**: Recurrent Neural Network (BiGRU)
  - **Dataset**: TESS
  - **Location**: `models/speech_pipeline/speech_only_model.pth` (production)

- **WavLM**: A pre-trained WavLM transformer model fine-tuned for emotion recognition.
  - **Model Type**: Transformer-based (pre-trained WavLM)
  - **Dataset**: TESS
  - **Location**: `archived_models/Speech_pipeline/wavlm_random_split.pth` (archived)

### Text Pipeline
Natural language processing models for emotion recognition from transcripts and text.

- **DistilBERT**: A distilled BERT transformer model for text classification.
  - **Model Type**: Transformer-based (DistilBERT)
  - **Dataset**: MELD (Multimodal EmotionLines Dataset)
  - **Location**: `archived_models/text_pipeline/bert_model.pth` (archived)

- **BiGRU**: A Bidirectional Gated Recurrent Unit network with embedding layer.
  - **Model Type**: Recurrent Neural Network (BiGRU)
  - **Dataset**: MELD
  - **Location**: `archived_models/text_pipeline/text_bigru_model.pth` (archived)

- **TF-IDF + Logistic Regression**: A scikit-learn baseline using TF-IDF vectorization with logistic regression.
  - **Model Type**: Classical Machine Learning (Bag-of-Words + Linear Classifier)
  - **Dataset**: TESS (transcripts)
  - **Location**: `models/text_pipeline/tess_text_model_OAF.joblib` (production)

### Fusion Pipeline
Late-fusion multimodal system combining audio and text predictions.

- **Late Fusion MLP**: A meta-classifier that combines speech and text model outputs.
  - **Model Type**: Feed-forward Neural Network (ensemble/fusion)
  - **Datasets**: TESS (combines audio model output + text model output)
  - **Location**: `models/fusion_pipeline/tess_fusion_model.pth` (production)

### Model Files Reference

| Pipeline | Model | Training Script | Test Script | Weights | Dataset | Status |
|----------|-------|-----------------|-------------|---------|---------|--------|
| Speech | MFCC + BiGRU | `train.py` | `test.py` | `speech_only_model.pth` | TESS | ✓ Production |
| Speech | WavLM | `train_wavlm.py` | `test_wavlm.py` | `wavlm_random_split.pth` | TESS | Archived |
| Text | DistilBERT | `train_bert.py` | `test_bert.py` | `bert_model.pth` | MELD | Archived |
| Text | TF-IDF + LR | `train.py` | `test.py` | `tess_text_model_OAF.joblib` | TESS | ✓ Production |
| Text | BiGRU | `train_bigru.py` | `test_bigru.py` | `text_bigru_model.pth` | MELD | Archived |
| Fusion | Late Fusion MLP | `train.py` | `test.py` | `tess_fusion_model.pth` | TESS | ✓ Production |

**Important:** Production models are in `models/` folder. Archived models are in `archived_models/` folder. Scripts in both locations have matching names but different paths. 

### Additional Required Artifacts

These supporting files must be present alongside the model weights for proper inference:

**Speech Pipeline**
- MFCC + BiGRU (production): No additional artifacts required (weights only)
- WavLM: No additional artifacts required (weights only)

**Text Pipeline**
- DistilBERT: `archived_models/text_pipeline/label_encoder.pkl`
  - Used to decode and map MELD emotion labels during evaluation
  
- BiGRU: 
  - `archived_models/text_pipeline/text_vocab.json` - Vocabulary mapping for text embeddings
  - `archived_models/text_pipeline/label_encoder_bigru.pkl` - Label encoder for emotion classes
  
- TF-IDF + Logistic Regression (production): No additional artifacts required (model contains all info)

**Fusion Pipeline**
- Late Fusion MLP (production): `models/fusion_pipeline/fusion_label_encoder.pkl`
  - Maps final probability outputs to emotion class labels

*Note: All of these supporting files are included alongside the weights in the Google Drive download link provided in the Quick Overview.*

## System Requirements

| Requirement | Details |
|-------------|----------|
| **Python Version** | Python 3.8 or higher (3.9+ recommended) |
| **CUDA Version** | 11.8+ (for GPU acceleration) |
| **RAM** | Minimum 8 GB; 16 GB+ recommended for training |
| **GPU** | NVIDIA GPU with CUDA support (optional but recommended for training) |
| **CPU** | Multi-core processor recommended |
| **Storage** | ~2 GB free space (after downloading datasets and models) |
| **Inference Runtime** | ~100-200ms per sample on GPU; ~1-2s per sample on CPU |
| **Training Runtime** | 10-30 minutes per model on GPU; several hours on CPU |

## Project Structure

```text
project/
├── archived_models/                  # Development and alternative model variants
│   ├── Results/                      # Archived evaluation results for alternative models
│   ├── Speech_pipeline/              # Archived speech models (e.g., WavLM variant)
│   └── text_pipeline/                # Archived text models (e.g., BERT, BiGRU)
├── data/                             # Raw datasets, ignored in Git
│   ├── MELD/                         # MELD CSV splits for fusion experiments
│   │   ├── dev_sent_emo.csv
│   │   ├── test_sent_emo.csv
│   │   ├── train_sent_emo_balanced.csv
│   │   └── train_sent_emo.csv
│   └── TESS Toronto emotional.../    # TESS speech dataset organized by emotion folders
├── models/                           # Production models (per project PDF)
│   ├── fusion_pipeline/              # Late-fusion system combining audio + text
│   │   ├── tess_fusion_model.pth
│   │   ├── fusion_label_encoder.pkl
│   │   ├── test.py
│   │   └── train.py
│   ├── speech_pipeline/              # MFCC + BiGRU speech model on TESS
│   │   ├── speech_only_model.pth
│   │   ├── test.py
│   │   └── train.py
│   └── text_pipeline/                # TF-IDF + LogReg text model on TESS
│       ├── tess_text_model_OAF.joblib
│       ├── test.py
│       └── train.py
├── Results/                          # Evaluation outputs for production models
│   ├── metrics/                      # Model performance metrics
│   └── plots/                        # Visualizations
├── data_loader.py                    # Generates tess_manifest.csv from TESS folder structure
├── README.md                         # Project documentation and quick start guide
├── tess_manifest.csv                 # TESS file paths and emotion labels
└── requirements.txt                  # Python dependencies
```

## Setup

1. Clone the repository.
  
  ```bash
  git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
  cd YOUR-REPOSITORY-NAME
  ```

2. Download and Extract External Assets
   
**Important:** Due to GitHub's file size limits, all heavy assets (model weights and datasets) are hosted externally on Google Drive. Download both files from the link below:
https://drive.google.com/drive/folders/16SWmrJ-QVp3ogzqmuLv9SByIkySN3pQL?usp=sharing

**Files to download:**
- `data.zip` - Contains MELD and TESS datasets
- `models.zip` - Contains production model weights (.pth, .joblib, .pkl) and supporting artifacts
- `archived_models.zip` - Contains archived/experimental model variants and their weights

**Extraction options:**

**Option A (Recommended - Automatic Merge):** 
Extract all three zip files directly into the root directory of your cloned repository. Your OS will prompt you to merge folders or overwrite. Click "Replace" or "Merge" to automatically place all files in their correct locations.

**Option B (Manual Placement):** 
Extract the zips into a temporary folder, then manually copy files into the repository using the Project Structure tree as your guide.

**Why this structure?**
- The `models/` and `archived_models/` folders in Git contain only Python training/testing scripts (`.py` files)
- Model weights (`.pth`, `.joblib`, `.pkl` files) are stored in the Google Drive zips to keep the Git repository size manageable
- This allows for easy collaboration while keeping the repo lightweight

3. Create a Virtual Environment

```powershell
# For Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

4. Install dependencies with `pip install -r requirements.txt`.

```powershell
# For Windows (PowerShell)
pip install -r requirements.txt
```

```bash
# For macOS/Linux
pip install -r requirements.txt
```

## Usage / How to Run the Code

**Important:** Before running any scripts, make sure your terminal is opened at the root directory of this repository and your virtual environment is activated.

### Production Models

#### Speech Pipeline

To evaluate the MFCC + BiGRU speech model on TESS:

```bash
python models/speech_pipeline/test.py
```

#### Text Pipeline

To evaluate the TF-IDF + Logistic Regression text model on TESS transcripts:

```bash
python models/text_pipeline/test.py
```

#### Fusion Pipeline

To evaluate the Late Fusion multimodal system:

```bash
python models/fusion_pipeline/test.py
```

### Archived/Alternative Models

To run archived model variants (WavLM, DistilBERT, BiGRU) from the project root:

```bash
# WavLM speech model (archived)
python archived_models/Speech_pipeline/test_wavlm.py

# DistilBERT text model (archived)
python archived_models/text_pipeline/test_bert.py

# BiGRU text model (archived)
python archived_models/text_pipeline/test_bigru.py
```

**Note on Training:** If you want to retrain any model from scratch, replace `test` with `train` in the commands above, for example `python models/speech_pipeline/train.py`. Training models may require a GPU for optimal performance.

## Results & Metrics

The following table outlines the training and testing accuracies across all six models evaluated in this project. Full classification reports, confusion matrices, and loss curves can be found in the `Results/` folder.
  
|Pipeline |       Model       |Training Accuracy | Testing Accuracy |
|---------|-------------------|------------------|------------------|
|Speech   |MFCC+BiGRU         |     99.52%       |    99.29%        |
|Speech   |WavLM              |     79.46%       |    85.71%        |
|Text     |TF-IDF+LR(Baseline)|     72.33%       |    14.29%        |
|Text     |BiGRU              |     79.65%       |    39.85%        |
|Text     |DistilBERT         |     66.89%       |    58.28%        |
|Fusion   |Late Fusion MLP    |     99.64%       |    99.29%        |

### Key Observations
1. **Dominance of Speech Acoustics:** MFCC+BiGRU and Fusion both achieved ~99% accuracy, demonstrating that speech acoustic features (prosody, tone, frequency) are the primary emotional carriers in the TESS dataset.
2. **Lexical Overfitting in Classical ML:** TF-IDF+LR showed a massive performance gap (72% train vs 14% test). This indicates the model memorized specific words in the training set which do not reliably correlate with emotions in these simple, repetitive transcripts.
3. **Efficiency of Handcrafted Features:** MFCC+BiGRU significantly outperformed WavLM. This suggests that for specialized datasets like TESS, traditional feature engineering (MFCCs) paired with recurrent architectures (BiGRU) can be more effective than large, general-purpose transformer models.
4. **Modality Performance Gap:** There is a stark contrast between speech (>99%) and text (~58% best) accuracy. This highlights that "what" is said in these recordings is far less informative for emotion recognition than "how" it is said.
5. **Robustness of Late Fusion:** The Late Fusion MLP successfully maintained the high accuracy of the speech pipeline (99.29%) without being degraded by the lower-performing text modality, proving the system's ability to prioritize the most reliable signal.

Note: For models that save the best checkpoint using validation/dev accuracy (e.g., BiGRU, DistilBERT), the training accuracy shown is taken from that best-checkpoint epoch, not the final logged epoch.

## Notes For Evaluators

- **Project Structure**: This repository implements the multimodal fusion architecture as specified in the project PDF. The three core models are stored in `models/` and follow the PDF requirements.

- **Production vs. Archived Models**: 
  - **Production models** (in `models/`): MFCC+BiGRU speech, TF-IDF+LR text, and Late Fusion MLP
  - **Archived/Alternative models** (in `archived_models/`): WavLM, DistilBERT, BiGRU, and experimental variants

- **Results Organization**:
  - `Results/` contains metrics and visualizations for the 3 production models
  - `archived_models/Results/` contains historical results for experimental variants (for reference only)

- **Model Weights & Assets**:
  - Production model weights must be downloaded from Google Drive (see Setup section)
  - Weights are not tracked in Git to keep the repository lightweight
  - All required supporting files (`.pkl`, `.joblib`) are included in the Google Drive downloads

- **Data Preprocessing**:
  - The balanced MELD file `data/MELD/train_sent_emo_balanced.csv` was generated by running `archived_models/text_pipeline/augment_data.py` on the original `data/MELD/train_sent_emo.csv`
  - **Do NOT re-run augmentation** - Running `augment_data.py` again will create duplicate samples in the already-balanced file
  - `tess_manifest.csv` is tracked in Git for reproducibility and can be regenerated using `data_loader.py` if needed

- **External Assets**: 
  - Download `data.zip`, `models.zip`, and `archived_models.zip` from the provided Google Drive link
  - Extract them into the root directory to populate datasets and model weights

## Authors

| Role | Name | Institution | Contact |
|------|------|-------------|---------|
| Student | Pranav Suresh| Saintgits College of Engineering | pranavsuresh1313@gmail.com|

## References

**Key Papers & Datasets Used:**

1. **TESS Dataset**: Pichora-Fuller, M. K., Dupuis, K. (2020). "Toronto Emotional Speech Set (TESS)". Mendeley Data, v1. Retrieved from: https://doi.org/10.17632/ngxh6rp47g.1

2. **MELD Dataset**: Zadeh, A. B., Tanveer, M. I., Morency, L. P. (2018). "MELD: A Multimodal EmotionLines Dataset for Emotion Recognition and Sentiment Analysis". Proceedings of ACL. Retrieved from: https://arxiv.org/abs/1810.02508

3. **WavLM**: Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., ... & Wei, F. (2021). "WavLM: Large-Scale Self-Supervised Pre-training for Speech". arXiv preprint arXiv:2110.13900.

4. **DistilBERT**: Sanh, V., Debut, L., Dernoncourt, F., Louf, R. (2020). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter". arXiv preprint arXiv:1910.01108.

**Tools & Libraries:**
- PyTorch: https://pytorch.org/
- scikit-learn: https://scikit-learn.org/
- librosa: https://librosa.org/
- Hugging Face Transformers: https://huggingface.co/transformers/
