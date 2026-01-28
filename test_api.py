# -*- coding: utf-8 -*-
import requests
import base64
import json
import os

API_URL = "http://localhost:8000/api/voice-detection"
API_KEY = "sk_test_voice_detection_123456789"

print("="*60)
print("TESTING VOICE DETECTION API")
print("="*60)

audio_file = "data/train/tamil/ai/ai_sample_tamil_01.mp3"

if not os.path.exists(audio_file):
    print("ERROR: File not found:", audio_file)
    exit(1)

print("\nTest: AI Sample (Tamil)")
print("File:", audio_file)

try:
    # Convert to Base64
    print("Converting to Base64...")
    with open(audio_file, 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    print("Base64 ready. Length:", len(audio_base64))
    
    # API Request
    payload = {
        "language": "Tamil",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    # Send with LONGER TIMEOUT (120 seconds = 2 minutes)
    print("\nSending to API (timeout: 120 seconds)...")
    response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
    
    # Response
    print("Status Code:", response.status_code)
    
    if response.status_code == 200:
        result = response.json()
        print("\nSUCCESS!")
        print("Prediction:", result['prediction'])
        print("Confidence:", result['confidence'])
        print("Processing Time:", result['processing_time_ms'], "ms")
    else:
        print("\nERROR:")
        print(response.text)

except requests.exceptions.Timeout:
    print("\nERROR: API took too long to respond (timeout)")
    print("The model processing is slow. This is normal for first request.")
    
except Exception as e:
    print("ERROR:", str(e))
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
