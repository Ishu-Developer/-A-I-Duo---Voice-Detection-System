cat > README.md << 'EOF'
# A-I Duo Voice Detection System

## Team
- Person A: Adarsh
- Person B: Ishu

## Project Description
AI-generated vs Human voice detection system supporting 5 languages:
- Tamil
- English
- Hindi
- Malayalam
- Telugu

## Model Performance
- Accuracy: 98.18%
- Confidence Range: 0.0 - 1.0
- Languages Supported: 5
- Processing Time: ~4 seconds per audio

## Deployment
- Local: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Testing
All endpoints tested and working:
- GET /health - Health check
- POST /api/voice-detection - Voice detection with Base64 audio

## API Usage
```bash
curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_voice_detection_123456789" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "[BASE64_AUDIO]"
  }'