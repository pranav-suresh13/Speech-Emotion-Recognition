# Speech Emotion Recognition

This project is a multimodal emotion recognition system that combines speech and text signals. It includes separate pipelines for speech, text, and late fusion, along with saved metrics and plots for evaluation.

## Quick Overview
If you are reviewing this project for the first time, the repository is structured around four key areas.

Important Note: Due to GitHub's file size limits, all heavy assets have been zipped and hosted externally. Please download both data.zip and models.zip from the link below and extract them into this repository's root directory:
https://drive.google.com/drive/folders/16SWmrJ-QVp3ogzqmuLv9SByIkySN3pQL?usp=sharing

- `data/`:Raw datasets (MELD and TESS) used for training and testing.
- `models/`:Training scripts, evaluation scripts, and the pre-trained weights/supporting artifacts for each pipeline.
- `Results/`:Evaluation outputs such as metrics tables, confusion matrices, and visualizations. 
- `requirements.txt`:Python dependencies required to run the code locally. 

## Pipelines & Models Overview

- **Speech pipeline**: 2 models trained on TESS (MFCC + MLP baseline and WavLM variant).
- **Text pipeline**: 3 models (DistilBERT and BiGRU trained on MELD, plus a TF-IDF + Logistic Regression baseline trained on TESS transcripts).
- **Fusion pipeline**: late-fusion system that combines the TESS audio model and TESS text model outputs. 

### Model Files Reference

| Pipeline|    Model      |Training Script |  Test Script  |          Weights             | 
|---------|---------------|----------------|---------------|------------------------------|
| Speech  |MFCC + MLP     |`train.py`      |`test.py`      | `speech_only_model.pth`      | 
| Speech  |WavLM          |`train_wavlm.py`|`test_wavlm.py`| `wavlm_random_split.pth`     | 
| Text    |DistilBERT     |`train_bert.py` |`test_bert.py` | `bert_model.pth`             | 
| Text    |TF-IDF + LR    |`train_tess.py` |`test_tess.py` | `tess_text_model_OAF.joblib` | 
| Text    |BiGRU          |`train_bigru.py`|`test_bigru.py`| `text_bigru_model.pth`       | 
| Fusion  |Late Fusion MLP|`train.py`      |`test.py`      | `tess_fusion_model.pth`      | 

Additional Required Artifacts:

The DistilBERT text model uses `label_encoder.pkl` to keep the MELD emotion labels consistent between training and evaluation.

The BiGRU text model requires `text_vocab.json` and `label_encoder_bigru.pkl` to decode the vocabulary and labels.

The Late Fusion MLP requires `fusion_label_encoder.pkl` to map the final probability outputs.
(Note: All of these supporting files are included alongside the weights in the Google Drive download link above).

## Project Structure

```text
project/
├── data/                             # Raw datasets, ignored in Git
│   ├── MELD/                         # MELD CSV splits for text and fusion experiments
│   │   ├── dev_sent_emo.csv          # Development split
│   │   ├── test_sent_emo.csv         # Test split
│   │   ├── train_sent_emo_balanced.csv # Balanced training split
│   │   └── train_sent_emo.csv        # Original training split
│   └── TESS Toronto emotional.../    # TESS speech dataset organized by emotion folders
├── models/                           # Training and evaluation pipelines
│   ├── fusion_pipeline/              # Late-fusion system using TESS audio features and TESS text predictions
│   │   ├── tess_fusion_model.pth     # Trained fusion MLP weights
│   │   ├── test.py                   # Tests the fusion model
│   │   └── train.py                  # Trains the fusion meta-classifier
│   ├── speech_pipeline/              # Speech emotion recognition system with MFCC + MLP and WavLM on TESS
│   │   ├── speech_only_model.pth     # MFCC + MLP speech model weights
│   │   ├── test.py                   # Tests the speech-only MFCC + MLP model
│   │   ├── test_wavlm.py             # Tests the WavLM variant
│   │   ├── train.py                  # Trains the speech-only MFCC + MLP model
│   │   ├── train_wavlm.py            # Trains the WavLM variant
│   │   └── wavlm_random_split.pth    # WavLM weights from a random split
│   └── text_pipeline/                # Text emotion recognition system with MELD models and a TESS baseline
│       ├── augment_data.py           # Text augmentation utilities
│       ├── bert_model.pth            # Fine-tuned DistilBERT weights
│       ├── label_encoder_bigru.pkl   # Label mapping for the BiGRU text model
│       ├── tess_text_model_OAF.joblib # TF-IDF + Logistic Regression baseline
│       ├── test_bert.py              # Tests the DistilBERT model
│       ├── test_bigru.py             # Tests the BiGRU text model
│       ├── test_tess.py              # Tests the TESS baseline model
│       ├── text_bigru_model.pth      # Trained BiGRU weights
│       ├── text_vocab.json           # Vocabulary mapping for BiGRU
│       ├── train_bert.py             # Trains the DistilBERT model
│       ├── train_bigru.py            # Trains the BiGRU text model
│       ├── train_tess.py             # Trains the TF-IDF + Logistic Regression baseline
│       └── visualize_attention.py    # Generates attention heatmaps
├── Results/                          # Saved outputs from experiments
│   ├── metrics/                      # CSV and TXT files with model performance
│   └── plots/                        # Confusion matrices, t-SNE plots, and training curves
├── data_loader.py                    # Scans TESS folders and generates tess_manifest.csv
├── README.md                         # Project documentation and quick start guide
├── tess_manifest.csv                 # TESS file-path and emotion index used by training/testing scripts
└── requirements.txt                  # Python dependencies
```

## Setup

1. Clone the repository.
  
  git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
  cd YOUR-REPOSITORY-NAME

2. Extract External Assets
Before running any code, ensure you have downloaded data.zip and models.zip from the Google Drive link provided in the Quick Overview. There are two ways to integrate these files into your cloned repository:

Option A (Recommended - Automatic Merge): Extract both zip files directly into the root directory of this repository. Because models.zip mirrors the repository's folder structure, your OS will ask if you want to merge folders or overwrite existing .py files. Click "Replace" or "Merge" to automatically drop the model weights perfectly alongside the cloned scripts.

Option B (Manual Placement): Extract the zips into a separate folder on your computer. Manually copy the .pth, .joblib, and .pkl files and paste them into their correct subfolders inside the repository, using the Project Structure tree above as your guide.

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

### Speech Pipeline

To evaluate the acoustic models on the TESS dataset:

```bash
# Run the baseline MFCC + MLP model
python models/speech_pipeline/test.py

# Run the advanced WavLM model
python models/speech_pipeline/test_wavlm.py
```

### Text Pipeline

To evaluate the natural language models on the MELD dataset:

```bash
# Run the best-performing DistilBERT model
python models/text_pipeline/test_bert.py

# Run the BiGRU architecture
python models/text_pipeline/test_bigru.py

# Run the TESS baseline (TF-IDF + Logistic Regression)
python models/text_pipeline/test_tess.py
```

### Fusion Pipeline

To evaluate the final multimodal Late Fusion meta-classifier:

```bash
python models/fusion_pipeline/test.py
```

**Note on Training:** If you want to retrain any model from scratch, replace `test` with `train` in the commands above, for example `python models/text_pipeline/train_bert.py`. Training the WavLM and DistilBERT models requires a GPU.

## Results & Metrics

The following table outlines the training and testing accuracies across all six models evaluated in this project. Full classification reports, confusion matrices, and loss curves can be found in the `Results/` folder.

|Pipeline |       Model       |Training Accuracy | Testing Accuracy |
|---------|-------------------|------------------|------------------|
|Speech   |MFCC+MLP(Baseline) |     99.52%       |    99.29%        |
|Speech   |WavLM              |     79.46%       |    85.71%        |
|Text     |TF-IDF+LR(Baseline)|     72.33%       |    14.29%        |
|Text     |BiGRU              |     65.66%       |    39.89%        |
|Text     |DistilBERT         |     57.22%       |    54.79%        |
|Fusion   |Late Fusion MLP    |     99.64%       |    99.29%        |

Note: For models that save the best checkpoint using validation/dev accuracy (e.g., BiGRU, DistilBERT), the training accuracy shown is taken from that best-checkpoint epoch, not the final logged epoch.

## Notes For Evaluators

- The data folder is intentionally excluded from Git because it contains large raw datasets.
- Trained model files are included in the workspace for convenience, but large artifacts may also be shared separately if needed.
- The script `models/text_pipeline/augment_data.py` was used during development to balance MELD, and evaluators do not need to run it.
- The balanced MELD file is `data/MELD/train_sent_emo_balanced.csv` (generated from `data/MELD/train_sent_emo.csv`); running augmentation again may duplicate samples.
- `tess_manifest.csv` is tracked in Git for reproducibility and is consumed by multiple speech/text/fusion scripts.
- `data_loader.py` can regenerate `tess_manifest.csv` from the TESS folder structure when needed.
- File comments in the structure above match the actual scripts and artifacts present in the workspace.
