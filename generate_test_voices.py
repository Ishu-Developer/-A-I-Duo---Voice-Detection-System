from gtts import gTTS
import os
import time

# ============================================
# CONFIGURATION
# ============================================
TEST_FOLDER = "data/test/ai_test_voice"

# Create test folder
os.makedirs(TEST_FOLDER, exist_ok=True)

# Language codes
LANG_CODES = {
    'tamil': 'ta',
    'english': 'en',
    'hindi': 'hi',
    'malayalam': 'ml',
    'telugu': 'te'
}

# Test texts (simple, short - for quick testing)
TEST_TEXTS = {
    'tamil': [
        'நிறைய வாழ்க',  # Long live
        'அறிவு வல்லமை',  # Knowledge is power
    ],
    'english': [
        'Hello world test',
        'This is artificial intelligence',
    ],
    'hindi': [
        'नमस्ते दुनिया',  # Hello world
        'कृत्रिम बुद्धिमत्ता',  # Artificial intelligence
    ],
    'malayalam': [
        'ആയാലെ വേൾഡ്',  # Hello world
        'കൃത്രിമ ബുദ്ധിമത്ത',  # Artificial intelligence
    ],
    'telugu': [
        'హలో వరల్డ్',  # Hello world
        'కృత్రిమ బుద్ధిమత్త',  # Artificial intelligence
    ]
}

# ============================================
# GENERATE TEST VOICES
# ============================================
print("=" * 70)
print("🎤 TEST VOICE GENERATOR - 2 SAMPLES PER LANGUAGE")
print("=" * 70)
print(f"\nTotal samples: 10 (2 per language)")
print(f"Languages: 5")
print(f"Save location: {TEST_FOLDER}/\n")

total_created = 0
errors = 0

for lang, texts in TEST_TEXTS.items():
    print(f"\n🎵 [{lang.upper()}] Generating 2 test samples...")
    print("-" * 70)
    
    lang_code = LANG_CODES[lang]
    
    for i, text in enumerate(texts, 1):
        try:
            # Generate TTS
            tts = gTTS(
                text=text,
                lang=lang_code,
                slow=False
            )
            
            # Save file
            filename = f"{TEST_FOLDER}/{lang}_test_ai_{i}.mp3"
            tts.save(filename)
            
            print(f"  ✅ Created: {filename}")
            total_created += 1
            
            # Small delay
            time.sleep(0.5)
        
        except Exception as e:
            errors += 1
            print(f"  ❌ Error: {str(e)[:50]}")

# ============================================
# SUMMARY & TESTING SCRIPT
# ============================================
print("\n" + "=" * 70)
print("🎉 TEST VOICES GENERATED!")
print("=" * 70)
print(f"\n📊 SUMMARY:")
print(f"  ✅ Created: {total_created} samples")
print(f"  ❌ Errors: {errors}")
print(f"\n📂 LOCATION:")
print(f"  {TEST_FOLDER}/")

print(f"\n📋 TEST FILES:")
for lang in LANG_CODES.keys():
    print(f"  • {lang.upper()}:")
    print(f"    - {TEST_FOLDER}/{lang}_test_ai_1.mp3")
    print(f"    - {TEST_FOLDER}/{lang}_test_ai_2.mp3")

print(f"\n" + "=" * 70)
print("🚀 QUICK TESTING GUIDE")
print("=" * 70)

print(f"""
STEP 1: Start server (Terminal 1)
  cd ~/-A-I-Duo---Voice-Detection-System
  source voice_env/Scripts/activate
  python main.py

STEP 2: Generate Base64 & Test (Terminal 2)
  python3 << 'EOF'
import base64
import requests
import os

API_URL = "http://localhost:8000/api/voice-detection"
API_KEY = "sk_test_voice_detection_123456789"

test_folder = "{TEST_FOLDER}"

# Get all test files
test_files = [f for f in os.listdir(test_folder) if f.endswith('.mp3')]

print(f"\\n🎯 Testing {{len(test_files)}} files...\\n")

for file in sorted(test_files):
    filepath = os.path.join(test_folder, file)
    
    # Extract language from filename (e.g., 'tamil_test_ai_1.mp3')
    lang = file.split('_')[0].title()
    
    try:
        # Convert to Base64
        with open(filepath, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Send request
        response = requests.post(
            API_URL,
            json={{
                "language": lang,
                "audioFormat": "mp3",
                "audioBase64": audio_base64
            }},
            headers={{"x-api-key": API_KEY}},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {{file}}")
            print(f"   Language: {{lang}}")
            print(f"   Prediction: {{result['prediction']}}")
            print(f"   Confidence: {{result['confidence']:.2%}}")
            print()
        else:
            print(f"❌ {{file}}: {{response.status_code}} - {{response.text}}")
    
    except Exception as e:
        print(f"❌ {{file}}: {{str(e)}}")

EOF

STEP 3: Browser Testing
  Open: http://localhost:8000/docs
  
  For each test file:
  1. Convert to Base64:
     python3 << 'EOF'
import base64
with open('data/test/ai_test_voice/[filename]', 'rb') as f:
    print(base64.b64encode(f.read()).decode('utf-8'))
EOF
  
  2. POST /api/voice-detection with:
     - language: [Tamil/English/Hindi/Malayalam/Telugu]
     - audioBase64: [paste Base64 string]
     - x-api-key: sk_test_voice_detection_123456789
  
  3. Verify:
     ✅ prediction: "AI_GENERATED"
     ✅ confidence: > 0.95

EXPECTED RESULTS:
  All 10 test files should be detected as "AI_GENERATED"
  with confidence > 0.95
""")

print("=" * 70)
print("✅ Test Setup Complete!")
print("=" * 70)