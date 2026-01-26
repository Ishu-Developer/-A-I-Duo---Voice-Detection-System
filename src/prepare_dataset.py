# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from audio_processor import AudioProcessor

class DatasetPreparer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.processor = AudioProcessor()
        
    def prepare_training_data(self):
        """Prepare training dataset with labels"""
        X_train = []
        y_train = []
        metadata = []
        
        # Directory structure: data/train/language/category/
        for lang_dir in sorted(self.data_dir.glob("train/*/")):
            language = lang_dir.name
            
            for category_dir in sorted(lang_dir.glob("*/")):
                category = category_dir.name  # 'ai' or 'human'
                label = 1 if category == 'ai' else 0
                
                # Find ALL audio files
                # Replace the file finding section with this:

                audio_files = list(set(
                    list(category_dir.glob("*.mp3")) +
                    list(category_dir.glob("*.MP3")) +
                    list(category_dir.glob("*.wav")) +
                    list(category_dir.glob("*.WAV"))
                ))
                print(f"Processing {language}/{category}: {len(audio_files)} files")
                
                for audio_file in audio_files:
                    try:
                        features = self.processor.extract_features(str(audio_file))
                        if features is not None and len(features) > 0:
                            X_train.append(features)
                            y_train.append(label)
                            metadata.append({
                                'file': audio_file.name,
                                'language': language,
                                'category': category,
                                'label': label
                            })
                            print(f"  ✓ {audio_file.name}")
                        else:
                            print(f"  ⚠️ {audio_file.name} - empty features")
                    except Exception as e:
                        print(f"  ✗ {audio_file.name}: {e}")
        
        if len(X_train) == 0:
            print("❌ No audio files found!")
            print("Check folder structure: data/train/[language]/[ai|human]/*.mp3")
            return None, None, None
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"\n📊 Before normalization: {X_train.shape}")
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Save prepared data
        np.save("data/X_train.npy", X_train_scaled)
        np.save("data/y_train.npy", y_train)
        pd.DataFrame(metadata).to_csv("data/metadata.csv", index=False)
        
        print(f"\n✅ Dataset prepared:")
        print(f"   Total samples: {len(X_train)}")
        print(f"   Feature dimension: {X_train.shape[1]}")
        print(f"   AI samples: {sum(y_train)}")
        print(f"   Human samples: {len(y_train) - sum(y_train)}")
        print(f"   Ratio: {sum(y_train)}/{len(y_train)} AI")
        print(f"\n📁 Files saved:")
        print(f"   - data/X_train.npy ({X_train_scaled.shape})")
        print(f"   - data/y_train.npy ({y_train.shape})")
        print(f"   - data/metadata.csv")
        
        return X_train_scaled, y_train, scaler

# Run preparation
if __name__ == "__main__":
    print("🎤 Preparing voice detection dataset...\n")
    preparer = DatasetPreparer()
    X_train, y_train, scaler = preparer.prepare_training_data()
