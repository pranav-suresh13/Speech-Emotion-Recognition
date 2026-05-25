import os
import pandas as pd
import nlpaug.augmenter.word as naw
import nltk


def _ensure_nltk_resource(resource_id, download_name=None):
    try:
        nltk.data.find(resource_id)
    except LookupError:
        nltk.download(download_name or resource_id.split("/")[-1], quiet=True)


# Ensure required NLTK data
_ensure_nltk_resource("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
_ensure_nltk_resource("corpora/wordnet", "wordnet")
_ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MELD_DIR = os.path.join(BASE_DIR, "data", "MELD")
ORIGINAL_CSV = os.path.join(MELD_DIR, "train_sent_emo.csv")
BALANCED_CSV = os.path.join(MELD_DIR, "train_sent_emo_balanced.csv")


def balance_dataset():
    print(f"Loading original dataset from: {ORIGINAL_CSV}")
    df = pd.read_csv(ORIGINAL_CSV)

    if "Utterance" in df.columns:
        df["text"] = df["Utterance"]
    if "Emotion" in df.columns:
        df["emotion"] = df["Emotion"]

    df["emotion"] = df["emotion"].str.lower()
    df["emotion"] = df["emotion"].replace({"anger": "angry", "joy": "happy", "sadness": "sad"})

    valid_emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    df = df[df["emotion"].isin(valid_emotions)].reset_index(drop=True)

    print("\nOriginal class distribution:")
    print(df["emotion"].value_counts())

    target_count = 500
    aug = naw.SynonymAug(aug_src="wordnet", aug_p=0.3)

    augmented_rows = []
    error_count = 0
    error_samples = []

    print("\nSynthesizing new data for minority classes...")
    for emotion in valid_emotions:
        class_df = df[df["emotion"] == emotion]
        current_count = len(class_df)

        if current_count < target_count:
            needed = target_count - current_count
            print(f"  Generating {needed} synthetic samples for '{emotion}'")
            samples_to_augment = class_df.sample(n=needed, replace=True, random_state=42)

            for _, row in samples_to_augment.iterrows():
                original_text = str(row["text"])
                try:
                    augmented = aug.augment(original_text)
                    if isinstance(augmented, list):
                        synthetic_text = augmented[0]
                    else:
                        synthetic_text = augmented
                    new_row = row.copy()
                    new_row["text"] = synthetic_text
                    augmented_rows.append(new_row)
                except Exception as exc:
                    error_count += 1
                    if len(error_samples) < 3:
                        error_samples.append(f"{type(exc).__name__}: {exc}")
                    continue

    if error_count > 0:
        print(f"\nAugmentation errors encountered: {error_count}")
        for sample in error_samples:
            print(f"  Example error: {sample}")

    if augmented_rows:
        df_synth = pd.DataFrame(augmented_rows)
        df_balanced = pd.concat([df, df_synth], ignore_index=True)
    else:
        df_balanced = df

    print("\nBalanced class distribution:")
    print(df_balanced["emotion"].value_counts())

    # --- NEW: DATA CLEANING BLOCK ---
    print("\nScrubbing encoding errors from text...")
    for col in ["text", "Utterance"]:
        if col in df_balanced.columns:
            # Cast to string first to prevent attribute errors on empty rows
            df_balanced[col] = df_balanced[col].astype(str)
            df_balanced[col] = df_balanced[col].str.replace('Â’', "'", regex=False)
            df_balanced[col] = df_balanced[col].str.replace('Â‘', "'", regex=False)
            df_balanced[col] = df_balanced[col].str.replace('Â“', '"', regex=False)
            df_balanced[col] = df_balanced[col].str.replace('Â”', '"', regex=False)
            df_balanced[col] = df_balanced[col].str.replace('Â—', "-", regex=False)
            df_balanced[col] = df_balanced[col].str.replace('Â', "", regex=False)
            df_balanced[col] = df_balanced[col].str.replace('â€™', "'", regex=False)
            df_balanced[col] = df_balanced[col].str.replace('â€¦', "...", regex=False)
    # --------------------------------

    # Save with utf-8 encoding to lock in the clean text
    df_balanced.to_csv(BALANCED_CSV, index=False, encoding='utf-8')
    print(f"\nClean, balanced dataset saved to: {BALANCED_CSV}")


if __name__ == "__main__":
    balance_dataset()