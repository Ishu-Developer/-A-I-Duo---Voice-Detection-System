# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')

from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import base64
import io
import joblib
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONSTANTS & CONFIG
# ============================================
MODEL_PATH = "models/voice_detector.pkl"
SCALER_PATH = "models/scaler.pkl"
VALID_API_KEY = "sk_test_voice_detection_123456789"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ============================================
# LOAD MODEL & SCALER
# ============================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load scaler (नया)
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler loaded successfully from {SCALER_PATH}")
    except Exception as e:
        print(f"⚠️  Scaler not found, will run without scaling: {e}")
        scaler = None
else:
    print(f"⚠️  Scaler not found at {SCALER_PATH}")
    scaler = None

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="A-I Duo Voice Detection API",
    version="1.0.0",
    description="Detect AI-generated vs Human voices in multiple languages"
)

# ============================================
# DATA MODELS
# ============================================
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

class VoiceDetectionResponse(BaseModel):
    status: str
    prediction: str
    confidence: float
    processing_time_ms: float
    request_id: str

# ============================================
# FEATURE EXTRACTION (Enhanced)
# ============================================
def extract_features(audio_data, sr=22050):
    """Extract 44 enhanced features from audio"""
    try:
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_data), sr=sr, duration=3)
        
        # Extract features
        features = []
        
        # 1. MFCC (13 coefficients + 13 std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))  # Enhanced
        
        # 2. Spectral Centroid (mean + std)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(spec_centroid))
        features.append(np.std(spec_centroid))
        
        # 3. Spectral Rolloff (mean + std)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(spec_rolloff))
        features.append(np.std(spec_rolloff))
        
        # 4. Zero Crossing Rate (mean + std)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # 5. Chroma Features (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        features.extend(np.mean(chroma, axis=1))
        
        # 6. RMS Energy (mean + std)
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # Ensure we have exactly 44 features
        features = np.array(features)
        
        if len(features) < 44:
            features = np.pad(features, (0, 44 - len(features)), mode='constant')
        else:
            features = features[:44]
        
        return features.reshape(1, -1)
    
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {str(e)}")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "A-I Duo Voice Detection API",
        "version": "1.0.0",
        "docs": "http://localhost:8000/docs",
        "endpoints": {
            "health": "GET /health",
            "detect_with_base64": "POST /api/voice-detection",
            "detect_with_file": "POST /api/voice-detection-file"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": "voice_detector_v2",
        "scaler": "enabled" if scaler else "disabled",
        "supported_languages": SUPPORTED_LANGUAGES
    }

@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(None)
):
    """
    METHOD 1: Base64 Encoded Audio
    
    Detect if audio is AI-generated or Human voice
    
    Parameters:
    - language: Tamil, English, Hindi, Malayalam, Telugu
    - audioFormat: mp3, wav, ogg
    - audioBase64: Base64 encoded audio file
    - x-api-key: API authentication key
    
    Returns:
    - prediction: AI_GENERATED or HUMAN
    - confidence: 0.0 to 1.0
    - processing_time_ms: Processing duration
    """
    
    import time
    start_time = time.time()
    request_id = base64.b64encode(os.urandom(12)).decode('utf-8')
    
    try:
        # Validate API Key
        if x_api_key != VALID_API_KEY:
            raise HTTPException(
                status_code=403,
                detail="Invalid API Key"
            )
        
        # Validate language
        if request.language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Language not supported. Use: {SUPPORTED_LANGUAGES}"
            )
        
        # Decode Base64
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Base64 encoding: {str(e)}"
            )
        
        # Extract features
        features = extract_features(audio_bytes)
        
        # Scale features (नया)
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        confidence = float(model.predict_proba(features_scaled).max())
        
        # Map prediction
        prediction_label = "AI_GENERATED" if prediction == 1 else "HUMAN"
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        return VoiceDetectionResponse(
            status="success",
            prediction=prediction_label,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            request_id=request_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

@app.post("/api/voice-detection-file", response_model=VoiceDetectionResponse)
async def detect_voice_file(
    file: UploadFile = File(...),
    language: str = "Tamil",
    x_api_key: str = Header(None)
):
    """
    METHOD 2: Direct File Upload
    
    Upload MP3/WAV/OGG file directly for detection
    
    Parameters:
    - file: Audio file (MP3, WAV, OGG)
    - language: Tamil, English, Hindi, Malayalam, Telugu
    - x-api-key: API authentication key
    
    Returns:
    - prediction: AI_GENERATED or HUMAN
    - confidence: 0.0 to 1.0
    - processing_time_ms: Processing duration
    """
    
    import time
    start_time = time.time()
    request_id = base64.b64encode(os.urandom(12)).decode('utf-8')
    
    try:
        # Validate API Key
        if x_api_key != VALID_API_KEY:
            raise HTTPException(
                status_code=403,
                detail="Invalid API Key"
            )
        
        # Validate language
        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Language not supported. Use: {SUPPORTED_LANGUAGES}"
            )
        
        # Validate file size
        file_size = 0
        audio_bytes = b""
        
        while chunk := await file.read(1024):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size: 10MB"
                )
            audio_bytes += chunk
        
        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )
        
        # Validate file extension
        filename = file.filename.lower()
        valid_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
        if not any(filename.endswith(ext) for ext in valid_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Supported: {valid_extensions}"
            )
        
        # Extract features
        features = extract_features(audio_bytes)
        
        # Scale features (नया)
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        confidence = float(model.predict_proba(features_scaled).max())
        
        # Map prediction
        prediction_label = "AI_GENERATED" if prediction == 1 else "HUMAN"
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        return VoiceDetectionResponse(
            status="success",
            prediction=prediction_label,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            request_id=request_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

# ============================================
# STARTUP & SHUTDOWN
# ============================================

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("🚀 A-I DUO VOICE DETECTION API STARTING")
    print("="*60)
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"✅ Scaler: {'Enabled' if scaler else 'Disabled'}")
    print(f"✅ Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
    print(f"✅ API Key: {VALID_API_KEY}")
    print("="*60)
    print("\n📍 ENDPOINTS:")
    print("  1. Base64 Method:  POST /api/voice-detection")
    print("  2. File Upload:    POST /api/voice-detection-file")
    print("  3. Health Check:   GET /health")
    print("  4. Swagger UI:     http://localhost:8000/docs")
    print("="*60 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    print("\n" + "="*60)
    print("API SHUTTING DOWN")
    print("="*60 + "\n")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("🚀 Starting A-I Duo Voice Detection API...")
    print("📍 Local: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🔑 Use API Key: sk_test_voice_detection_123456789")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )