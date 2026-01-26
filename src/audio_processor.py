import librosa
import numpy as np
from pathlib import Path

class AudioProcessor:
    def __init__(self, sr=16000):
        self.sr = sr
        
    def normalize_audio(self, audio_path):
        """Load and normalize audio to standard format"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
        
        # Normalize amplitude
        if len(y) == 0:
            return None
            
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        # Pad or trim to 5 seconds
        target_length = self.sr * 5
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
            
        return y
    
    def extract_features(self, audio_path):
        """Extract comprehensive audio features"""
        y = self.normalize_audio(audio_path)
        if y is None:
            return None
        
        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        spec_cent_mean = np.mean(spec_cent)
        spec_cent_std = np.std(spec_cent)
        
        # Spectral Rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        spec_rolloff_mean = np.mean(spec_rolloff)
        spec_rolloff_std = np.std(spec_rolloff)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Combine all features (40 total)
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            [zcr_mean, zcr_std],
            [spec_cent_mean, spec_cent_std],
            [spec_rolloff_mean, spec_rolloff_std],
            chroma_mean
        ])
        
        return features
