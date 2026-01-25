# -A-I-Duo---Voice-Detection-System
The Voice Detection System with team A-I Duo (Adarsh-Ishu) is an AI-powered tool for the India AI Impact Buildathon. <br>
It classifies audio inputs as AI-generated or human-generated, supporting multi-language detection in Tamil, English, Hindi, Malayalam, and Telugu.

# Voice Detection System (AI-Generated vs Human)

## Overview
AI-Generated Voice Detection (Multi-Language) system for detecting synthetic voices in Tamil, English, Hindi, Malayalam, Telugu.  
**Target**: ~80% accuracy API jo audio input le aur JSON output de (e.g., {"voice_type": "AI", "confidence": 0.85}).  
Team: **A-I Duo** (Adarsh & Ishu) – India AI Impact Buildathon entry.

## Features
- Multi-language support (Indian languages).
- Audio preprocessing aur ML model (e.g., MFCC features, CNN/SVM).
- FastAPI-based endpoint for easy deployment.
- Demo with sample audios.

## Tech Stack
| Category | Tools                        |
|----------|------------------------------|
| Language | Python 3.9+                  |
| Framework | FastAPI, PyTorch/TensorFlow |
| Audio | Librosa, torchaudio             |
| ML | Scikit-learn, custom CNN           |
