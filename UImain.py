import sys
sys.path.insert(0, 'src')

from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import base64
import io
import joblib
import numpy as np
import librosa
import warnings
import traceback
warnings.filterwarnings('ignore')

# ============================================
# CONSTANTS & CONFIG
# ============================================
MODEL_PATH = "models/voice_detector.pkl"
SCALER_PATH = "models/scaler.pkl"
VALID_API_KEY = "voice_test_2026"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ============================================
# LOAD MODEL & SCALER (ENHANCED WITH DEBUGGING)
# ============================================
print("\n" + "="*60)
print("🔍 LOADING MODEL AND SCALER")
print("="*60)

# Debug: Show current directory and files
print(f"📂 Current directory: {os.getcwd()}")
print(f"📂 Files in current directory: {os.listdir('.')}")
if os.path.exists('models'):
    print(f"📂 Files in models/: {os.listdir('models')}")
else:
    print("⚠️  WARNING: 'models' directory not found!")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ CRITICAL ERROR: Model file not found at {MODEL_PATH}")
    print(f"   Expected path: {os.path.abspath(MODEL_PATH)}")
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
else:
    print(f"✅ Model file found: {MODEL_PATH}")

# Load model with comprehensive error handling
model = None
try:
    print(f"   Loading model...")
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
    print(f"   Model type: {type(model)}")
    print(f"   Model has predict: {hasattr(model, 'predict')}")
    print(f"   Model has predict_proba: {hasattr(model, 'predict_proba')}")
except Exception as e:
    print(f"❌ CRITICAL: Error loading model!")
    print(f"   Exception: {e}")
    print(f"   Exception type: {type(e).__name__}")
    traceback.print_exc()
    model = None

# Load scaler
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        print(f"   Loading scaler...")
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler loaded successfully!")
        print(f"   Scaler type: {type(scaler)}")
    except Exception as e:
        print(f"⚠️  Warning: Scaler loading error: {e}")
        scaler = None
else:
    print(f"⚠️  Scaler not found at {SCALER_PATH}, continuing without scaler")

print("="*60)
print(f"📊 Final Status:")
print(f"   Model loaded: {model is not None}")
print(f"   Scaler loaded: {scaler is not None}")
print("="*60 + "\n")

# Validate model is not None before starting server
if model is None:
    print("❌ CRITICAL: Model failed to load!")
    print("   The application CANNOT start without a valid model.")
    raise RuntimeError("Model loading failed - application cannot start")

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="A-I Duo Voice Detection API",
    version="2.0.1",
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
# FEATURE EXTRACTION (Enhanced + FIXED)
# ============================================
def extract_features(audio_data, sr=22050):
    """Extract 44 enhanced features from audio"""
    try:
        # Create BytesIO object and reset pointer
        audio_buffer = io.BytesIO(audio_data)
        audio_buffer.seek(0)
        
        # Load audio from BytesIO buffer
        y, sr = librosa.load(audio_buffer, sr=sr, duration=3)
        
        # Extract features
        features = []
        
        # 1. MFCC (13 coefficients + 13 std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        
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
        
        # Ensure exactly 44 features
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

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML dashboard"""
    try:
        if os.path.exists("index.html"):
            with open("index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(
                content="""
                <html>
                <head><title>Frontend Not Found</title></head>
                <body style="font-family: Arial; padding: 40px; text-align: center;">
                    <h1>⚠️ Frontend Dashboard Not Found</h1>
                    <p>The index.html file is missing from the deployment.</p>
                    <p><a href="/docs" style="color: #3b82f6; text-decoration: none;">Visit API Documentation</a></p>
                </body>
                </html>
                """,
                status_code=404
            )
    except Exception as e:
        return HTMLResponse(
            content=f"""
            <html>
            <head><title>Error</title></head>
            <body style="font-family: Arial; padding: 40px; text-align: center;">
                <h1>❌ Error Loading Frontend</h1>
                <p>Error: {str(e)}</p>
                <p><a href="/docs" style="color: #3b82f6; text-decoration: none;">Visit API Documentation</a></p>
            </body>
            </html>
            """,
            status_code=500
        )

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model": "voice_detector_v2",
        "model_loaded": model is not None,
        "scaler_enabled": scaler is not None,
        "languages_supported": SUPPORTED_LANGUAGES,
        "version": "2.0.1"
    }

@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(None)
):
    """
    METHOD 1: Base64 Encoded Audio
    
    Detect if audio is AI-generated or Human voice
    """
    
    import time
    start_time = time.time()
    request_id = base64.b64encode(os.urandom(12)).decode('utf-8')
    
    try:
        # ✅ Model validation check
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service temporarily unavailable."
            )
        
        # Validate API Key
        if x_api_key != VALID_API_KEY:
            raise HTTPException(
                status_code=403,
                detail="Invalid API Key"
            )
        
        # Case-insensitive language validation
        language_normalized = request.language.strip().title()
        
        if language_normalized not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Language '{request.language}' not supported. Use: {SUPPORTED_LANGUAGES}"
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
        
        # Scale features
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
    """
    
    import time
    start_time = time.time()
    request_id = base64.b64encode(os.urandom(12)).decode('utf-8')
    
    try:
        # ✅ Model validation check
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service temporarily unavailable."
            )
        
        # Validate API Key
        if x_api_key != VALID_API_KEY:
            raise HTTPException(
                status_code=403,
                detail="Invalid API Key"
            )
        
        # Case-insensitive language validation
        language_normalized = language.strip().title()
        
        if language_normalized not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Language '{language}' not supported. Use: {SUPPORTED_LANGUAGES}"
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
        
        # Scale features
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
    print("🚀 A-I DUO VOICE DETECTION API v2.0.1 STARTING")
    print("="*60)
    print(f"✅ Model: {MODEL_PATH}")
    print(f"✅ Model Status: {'Loaded ✓' if model else 'Not Loaded ✗'}")
    print(f"✅ Scaler: {'Enabled ✓' if scaler else 'Disabled ✗'}")
    print(f"✅ Supported Languages: {', '.join(SUPPORTED_LANGUAGES)}")
    print(f"🔑 API Key: {VALID_API_KEY}")
    print("="*60)
    print("\n📍 ENDPOINTS:")
    print("  1. Frontend:       GET  /")
    print("  2. Health Check:   GET  /health")
    print("  3. Base64 Method:  POST /api/voice-detection")
    print("  4. File Upload:    POST /api/voice-detection-file")
    print("  5. Swagger UI:     GET  /docs")
    print("="*60 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    print("\n" + "="*60)
    print("🛑 API SHUTTING DOWN")
    print("="*60 + "\n")

# ============================================
# MAIN - RAILWAY COMPATIBLE
# ============================================

if __name__ == "__main__":
    # Get PORT from environment variable (Railway sets this automatically)
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 Starting A-I Duo Voice Detection API v2.0.1...")
    print(f"📍 Host: 0.0.0.0")
    print(f"📍 Port: {port}")
    print(f"📚 Docs: /docs")
    print(f"🔑 API Key: {VALID_API_KEY}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
