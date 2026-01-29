import os
import joblib
import numpy as np
import librosa
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import io

warnings.filterwarnings('ignore')

def extract_features(audio_data, sr=22050):
    """Enhanced feature extraction"""
    try:
        if isinstance(audio_data, str):
            y, sr = librosa.load(audio_data, sr=sr, duration=3)
        else:
            y, sr = librosa.load(io.BytesIO(audio_data), sr=sr, duration=3)
        
        features = []
        
        # MFCC (13 + 13 std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        
        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(spec_centroid))
        features.append(np.std(spec_centroid))
        
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(spec_rolloff))
        features.append(np.std(spec_rolloff))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # Chroma (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        features = np.array(features)
        
        # Pad to 44 features
        if len(features) < 44:
            features = np.pad(features, (0, 44 - len(features)), mode='constant')
        else:
            features = features[:44]
        
        return features.reshape(1, -1)
    
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {str(e)}")


# ============================================
# LOAD DATA
# ============================================
print("=" * 70)
print("ADVANCED MODEL TRAINING v2")
print("=" * 70)

X = []
y = []

# Load AI voices
for lang in ['tamil', 'english', 'hindi', 'malayalam', 'telugu']:
    ai_folder = f"data/train/{lang}/ai"
    
    if os.path.exists(ai_folder):
        files = [f for f in os.listdir(ai_folder) if f.endswith(('.mp3', '.wav'))]
        print(f"\nLoading {lang.upper()} AI files: {len(files)} samples")
        
        for file in files:
            try:
                filepath = os.path.join(ai_folder, file)
                features = extract_features(filepath)
                X.append(features)
                y.append(1)  # AI
            except:
                pass

# Load Human voices
for lang in ['tamil', 'english', 'hindi', 'malayalam', 'telugu']:
    human_folder = f"data/train/{lang}/human"
    
    if os.path.exists(human_folder):
        files = [f for f in os.listdir(human_folder) if f.endswith(('.mp3', '.wav'))]
        print(f"Loading {lang.upper()} HUMAN files: {len(files)} samples")
        
        for file in files:
            try:
                filepath = os.path.join(human_folder, file)
                features = extract_features(filepath)
                X.append(features)
                y.append(0)  # Human
            except:
                pass

X = np.vstack(X)
y = np.array(y)

print(f"\nDataset loaded:")
print(f"   Features shape: {X.shape}")
print(f"   AI (1): {(y == 1).sum()}, Human (0): {(y == 0).sum()}")

# ============================================
# PREPROCESSING
# ============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# ============================================
# TRAINING WITH BETTER HYPERPARAMETERS
# ============================================
print(f"\nTraining XGBoost with optimized parameters...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================
# EVALUATION
# ============================================
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"\nModel Performance:")
print(f"   Training Accuracy: {train_acc:.4f}")
print(f"   Test Accuracy: {test_acc:.4f}")
print(f"   ROC-AUC Score: {roc_auc:.4f}")

# Detailed report
y_pred = model.predict(X_test)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['HUMAN', 'AI_GENERATED']))

# ============================================
# CROSS-VALIDATION
# ============================================
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"\nCross-Validation (5-fold):")
print(f"   Fold scores: {cv_scores}")
print(f"   Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================
# SAVE MODEL & SCALER
# ============================================
joblib.dump(model, "models/voice_detector.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print(f"\nModel saved: models/voice_detector.pkl")
print(f"Scaler saved: models/scaler.pkl")

print("\n" + "=" * 70)
print("ADVANCED TRAINING COMPLETE!")
print("=" * 70)