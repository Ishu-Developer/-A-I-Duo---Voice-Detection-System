# -*- coding: utf-8 -*-
import os
import base64
import io
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import librosa
import warnings
warnings.filterwarnings('ignore')

# Load trained model
MODEL_PATH = "models/voice_detector.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# API Configuration
VALID_API_KEYS = [
    "sk_test_voice_detection_123456789",  # Change this!
    "sk_prod_voice_detection_abcdefghijk"
]

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# FastAPI app
app = FastAPI(
    title="A-I Duo Voice Detection API",
    description="Detect AI-generated vs Human voices",
    version="1.0.0"
)

# Request model
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# Feature extraction (same as training)
class AudioProcessor:
    def __init__(self, sr=16000):
        self.sr = sr
        
    def normalize_audio(self, audio_data):
        """Normalize audio"""
        y = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
        
        # Pad or trim to 5 seconds
        target_length = self.sr * 5
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        return y
    
    def extract_features(self, audio_data):
        """Extract 44 features"""
        y = self.normalize_audio(audio_data)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # ZCR
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
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Combine features
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            [zcr_mean, zcr_std],
            [spec_cent_mean, spec_cent_std],
            [spec_rolloff_mean, spec_rolloff_std],
            chroma_mean
        ])
        
        return features

processor = AudioProcessor()

# Explanation generator
def get_explanation(prediction, confidence):
    """Generate human-readable explanation"""
    if prediction == 1:  # AI_GENERATED
        if confidence > 0.9:
            return "Strong synthetic voice characteristics detected: artificial pitch, unnatural prosody patterns"
        else:
            return "Synthetic voice patterns detected with moderate confidence"
    else:  # HUMAN
        if confidence > 0.95:
            return "Natural human voice detected with high confidence"
        else:
            return "Human voice detected with natural speech characteristics"

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": "voice_detector_v1",
        "supported_languages": SUPPORTED_LANGUAGES
    }

# Main API endpoint
@app.post("/api/voice-detection")
async def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(None)
):
    """
    Detect AI-generated vs Human voice
    
    Request:
    {
        "language": "Tamil",
        "audioFormat": "mp3",
        "audioBase64": "SUQzBAA..."
    }
    
    Response:
    {
        "status": "success",
        "language": "Tamil",
        "classification": "AI_GENERATED",
        "confidenceScore": 0.91,
        "explanation": "..."
    }
    """
    
    try:
        # 1. Validate API Key
        if not x_api_key or x_api_key not in VALID_API_KEYS:
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "message": "Invalid API key or malformed request"
                }
            )
        
        # 2. Validate language
        if request.language not in SUPPORTED_LANGUAGES:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Language not supported. Supported: {SUPPORTED_LANGUAGES}"
                }
            )
        
        # 3. Validate audio format
        if request.audioFormat != "mp3":
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Only MP3 format supported"
                }
            )
        
        # 4. Decode Base64
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
            audio_io = io.BytesIO(audio_bytes)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Invalid Base64 encoding: {str(e)}"
                }
            )
        
        # 5. Load and process audio
        try:
            y, sr = librosa.load(audio_io, sr=16000)
            if len(y) == 0:
                raise ValueError("Empty audio")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Failed to process audio: {str(e)}"
                }
            )
        
        # 6. Extract features
        try:
            features = processor.extract_features(y)
            features = features.reshape(1, -1)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Feature extraction failed: {str(e)}"
                }
            )
        
        # 7. Run prediction
        try:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            confidence = max(probability)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Prediction failed: {str(e)}"
                }
            )
        
        # 8. Format response
        classification = "AI_GENERATED" if prediction == 1 else "HUMAN"
        explanation = get_explanation(prediction, confidence)
        
        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": float(round(confidence, 4)),
            "explanation": explanation
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }
        )

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "A-I Duo Voice Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detect": "/api/voice-detection",
            "docs": "/docs"
        }
    }