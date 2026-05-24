import os
import pandas as pd

# Change this line to your exact nested folder path
DATA_DIR = 'data/TESS Toronto emotional speech set data'

def create_manifest():
    """Generate tess_manifest.csv by scanning TESS emotion folders"""
    rows = []
    
    # Define emotion mapping to normalize inconsistent folder names
    emotion_mapping = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'Fear': 'fear',
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'Sad': 'sad',
        'pleasant_surprise': 'surprise',
        'Pleasant_surprise': 'surprise',
        'pleasant_surprised': 'surprise',
    }
    
    # Iterate through emotion folders
    for emotion_folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, emotion_folder)
        
        if os.path.isdir(folder_path):
            # Extract emotion from folder name (e.g., 'OAF_angry' -> 'angry')
            extracted_emotion = emotion_folder.split('_', 1)[1] if '_' in emotion_folder else emotion_folder
            
            # Normalize emotion using mapping
            emotion = emotion_mapping.get(extracted_emotion, extracted_emotion.lower())
            
            # Scan for audio files
            for audio_file in os.listdir(folder_path):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(folder_path, audio_file)
                    rows.append({'path': file_path, 'emotion': emotion})
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv('tess_manifest.csv', index=False)
    
    print(f"✓ Manifest created successfully!")
    print(f"✓ Total entries mapped: {len(df)}")
    print(f"✓ File saved to: tess_manifest.csv")
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts())
    print(f"\nUnique emotions ({len(df['emotion'].unique())}): {sorted(df['emotion'].unique())}")

if __name__ == "__main__":
    create_manifest()
