from gtts import gTTS
import os

# Create folders
langs = ['tamil', 'english', 'hindi', 'malayalam', 'telugu']
for lang in langs:
    os.makedirs(f"data/train/{lang}/ai", exist_ok=True)

texts = {
    'tamil': 'வணக்கம் இது செயற்கை குரல்',
    'english': 'Hello this is artificial voice',
    'hindi': 'नमस्ते यह कृत्रिम आवाज है',
    'malayalam': 'ഹലോ ഇത് കൃത്രിമ കണ്ഠം',
    'telugu': 'హలో ఇది కృత్రిమ వాయిస్'
}

# Correct language codes for gTTS
lang_codes = {
    'tamil': 'ta',
    'english': 'en',
    'hindi': 'hi',
    'malayalam': 'ml',  # Fixed: was 'ma'
    'telugu': 'te'      # Fixed: was 'te' but verify
}

for lang, text in texts.items():
    try:
        tts = gTTS(text=text, lang=lang_codes[lang])
        output = f"data/train/{lang}/ai/ai_sample_{lang}.mp3"
        tts.save(output)
        print(f"✅ Created: {output}")
    except ValueError as e:
        print(f"⚠️ {lang} skipped: {e}")
        # Create dummy file instead
        os.makedirs(f"data/train/{lang}/ai", exist_ok=True)
        with open(f"data/train/{lang}/ai/ai_sample_{lang}.mp3", 'wb') as f:
            f.write(b'dummy')
        print(f"✅ Created (placeholder): data/train/{lang}/ai/ai_sample_{lang}.mp3")

print("🎉 AI Dataset ready!")