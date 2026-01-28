import requests
import base64
import json
import os

API_URL = "http://localhost:8000/api/voice-detection"
API_KEY = "sk_test_voice_detection_123456789"

# Test files
tests = [
    ("AI Sample (Tamil)", "data/train/tamil/ai/ai_sample_tamil_01.mp3", "Tamil"),
]

for test_name, audio_file, lang in tests:
    print(f"\n{'='*60}")
    print(f"Ìæ§ Test: {test_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(audio_file):
        print(f"‚ùå File not found: {audio_file}")
        continue
    
    try:
        # Convert to Base64
        with open(audio_file, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # API Request
        payload = {
            "language": lang,
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": API_KEY
        }
        
        # Send
        print(f"Ì≥§ Sending {len(audio_base64)/1024:.1f} KB of audio data...")
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        # Response
        if response.status_code == 200:
            result = response.json()
            print(f"\n‚úÖ SUCCESS (Status: 200)")
            print(f"\nÌ≥ä Results:")
            print(f"   ÌæØ Prediction: {result['prediction']}")
            print(f"   Ì≥à Confidence: {result['confidence']:.2%}")
            print(f"   ‚è±Ô∏è  Processing: {result['processing_time_ms']}ms")
        else:
            print(f"\n‚ùå Error (Status: {response.status_code})")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"‚ùå Error: {e}")

print(f"\n{'='*60}")
print("‚úÖ TESTING COMPLETE!")
print(f"{'='*60}")
