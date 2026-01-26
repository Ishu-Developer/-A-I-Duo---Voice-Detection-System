from google.cloud import texttospeech
import os

client = texttospeech.TextToSpeechClient()

# Sample texts in each language
texts = {
    'tamil': 'வணக்கம், இது ஒரு செயற்கை குரல் மாதிரி',
    'english': 'Hello, this is an artificial voice sample',
    'hindi': 'नमस्ते, यह एक कृत्रिम आवाज का नमूना है',
    'malayalam': 'ഹലോ, ഇത് ഒരു കൃത്രിമ വോയിസ് സാമ്പിൾ ആണ്',
    'telugu': 'హలో, ఇది ఆర్టిఫిషియల్ వాయిస్ నమూనా'
}

for lang, text in texts.items():
    # Generate and save audio
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code=f"{lang}-IN",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
    )
    
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    # Save audio
    with open(f"data/train/{lang}/ai/sample_{lang}.mp3", "wb") as out:
        out.write(response.audio_content)
    
    print(f"Generated {lang} AI voice")
