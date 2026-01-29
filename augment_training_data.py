# -*- coding: utf-8 -*-

import librosa
import soundfile as sf
import os
import numpy as np
from pathlib import Path

print("=" * 70)
print("��� DATA AUGMENTATION - ENHANCING AI VOICE DATASET")
print("=" * 70)

AI_VOICE_FOLDER = "data/train"
AUGMENTED_FOLDER = "data/train_augmented"

os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# Augmentation techniques
def pitch_shift(y, sr, n_steps):
    """Pitch को बदलो (+/- semitones)"""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate):
    """Speed को बदलो"""
    return librosa.effects.time_stretch(y, rate=rate)

def add_noise(y, noise_factor=0.005):
    """Slight noise add करो"""
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return augmented

def process_audio(file_path, lang, index):
    """एक file को load करके 3 variations बनाओ"""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Original save करो
        output_path_1 = f"{AUGMENTED_FOLDER}/{lang}_ai_{index}_orig.wav"
        sf.write(output_path_1, y, sr)
        
        # Pitch shift (+2 semitones)
        y_pitch = pitch_shift(y, sr, n_steps=2)
        output_path_2 = f"{AUGMENTED_FOLDER}/{lang}_ai_{index}_pitch.wav"
        sf.write(output_path_2, y_pitch, sr)
        
        # Time stretch (1.1x faster)
        y_time = time_stretch(y, rate=1.1)
        output_path_3 = f"{AUGMENTED_FOLDER}/{lang}_ai_{index}_fast.wav"
        sf.write(output_path_3, y_time, sr)
        
        # Noise (very little)
        y_noise = add_noise(y, noise_factor=0.003)
        output_path_4 = f"{AUGMENTED_FOLDER}/{lang}_ai_{index}_noise.wav"
        sf.write(output_path_4, y_noise, sr)
        
        print(f"✅ {os.path.basename(file_path)} - 4 variations created")
        return 4
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        return 0

# Process सभी AI files
total = 0
for lang in ['tamil', 'english', 'hindi', 'malayalam', 'telugu']:
    ai_folder = f"{AI_VOICE_FOLDER}/{lang}/ai"
    
    if os.path.exists(ai_folder):
        files = [f for f in os.listdir(ai_folder) if f.endswith(('.mp3', '.wav'))]
        print(f"\n��� [{lang.upper()}] Processing {len(files)} files...")
        
        for idx, file in enumerate(files[:5], 1):  # पहली 5 files
            filepath = os.path.join(ai_folder, file)
            created = process_audio(filepath, lang, idx)
            total += created

print("\n" + "=" * 70)
print(f"✅ AUGMENTATION COMPLETE!")
print(f"   Created: {total} augmented audio files")
print(f"   Location: {AUGMENTED_FOLDER}/")
print("=" * 70)

