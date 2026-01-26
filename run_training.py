# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')

from train_model import VoiceDetectionModel
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    print("🎤 Starting Day 3: Model Training\n")
    
    # Load data
    trainer = VoiceDetectionModel()
    X_train, y_train = trainer.load_data()
    
    # Split
    X_train_split, X_test, y_train_split, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"📂 Data split:")
    print(f"   Training: {X_train_split.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples\n")
    
    # Train
    trainer.train_xgboost(X_train_split, y_train_split)
    
    # Evaluate
    test_acc, roc_auc = trainer.evaluate(X_train_split, y_train_split, X_test, y_test)
    
    # Cross-validate
    trainer.cross_validate(X_train_split, y_train_split, cv=5)
    
    # Save
    trainer.save_model()
    
    print("\n✅ MODEL TRAINING COMPLETE!")
    print("✅ Model saved to: models/voice_detector.pkl")