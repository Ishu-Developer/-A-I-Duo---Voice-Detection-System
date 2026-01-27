import sys
sys.path.insert(0, 'src')

import uvicorn
from voice_api import app

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