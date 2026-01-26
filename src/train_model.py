# -*- coding: utf-8 -*-
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class VoiceDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.history = {}
        
    def load_data(self):
        """Load preprocessed training data"""
        X_train = np.load("data/X_train.npy")
        y_train = np.load("data/y_train.npy")
        
        print(f"📊 Dataset loaded:")
        print(f"   Features shape: {X_train.shape}")
        print(f"   Labels shape: {y_train.shape}")
        print(f"   AI (1): {sum(y_train)}, Human (0): {len(y_train) - sum(y_train)}")
        
        return X_train, y_train
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model with optimization"""
        
        # Handle class imbalance
        scale_pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-8)
        
        self.model = XGBClassifier(
            n_estimators=200,           # Number of trees
            max_depth=6,                # Tree depth
            learning_rate=0.1,          # Learning rate
            subsample=0.8,              # Row sampling
            colsample_bytree=0.8,       # Feature sampling
            scale_pos_weight=scale_pos_weight,  # Handle AI/Human imbalance
            random_state=42,
            n_jobs=-1,                  # Use all CPU cores
            verbosity=0
        )
        
        print(f"\n🤖 Training XGBoost model...")
        print(f"   Handling class imbalance (scale_pos_weight={scale_pos_weight:.2f})")
        
        self.model.fit(X_train, y_train, verbose=False)
        
        print(f"✅ Model training complete!")
        
        return self.model
    
    def evaluate(self, X_train, y_train, X_test, y_test):
        """Evaluate model performance"""
        
        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        # Test accuracy
        test_pred = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Probability predictions for ROC-AUC
        test_proba = self.model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, test_proba)
        
        print(f"\n📈 Model Performance:")
        print(f"   Training Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   ROC-AUC Score: {roc_auc:.4f}")
        
        print(f"\n📊 Classification Report (Test Set):")
        print(classification_report(
            y_test, test_pred,
            target_names=['HUMAN', 'AI_GENERATED'],
            digits=4
        ))
        
        print(f"\n🎯 Confusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, test_pred)
        print(f"   True Negatives: {cm[0,0]}")
        print(f"   False Positives: {cm[0,1]}")
        print(f"   False Negatives: {cm[1,0]}")
        print(f"   True Positives: {cm[1,1]}")
        
        self.history = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'test_predictions': test_pred,
            'test_probabilities': test_proba
        }
        
        return test_acc, roc_auc
    
    def save_model(self, model_path="models/voice_detector.pkl"):
        """Save trained model"""
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"\n💾 Model saved: {model_path}")
    
    def cross_validate(self, X_train, y_train, cv=5):
        """Perform cross-validation"""
        print(f"\n🔄 Performing {cv}-fold cross-validation...")
        
        scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        print(f"   Fold scores: {scores}")
        print(f"   Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("🎤 A-I DUO VOICE DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    trainer = VoiceDetectionModel()
    X_train, y_train = trainer.load_data()
    
    # Split data (80/20)
    X_train_split, X_test, y_train_split, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n📂 Data split:")
    print(f"   Training: {X_train_split.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    
    # Train model
    trainer.train_xgboost(X_train_split, y_train_split)
    
    # Evaluate
    test_acc, roc_auc = trainer.evaluate(X_train_split, y_train_split, X_test, y_test)
    
    # Cross-validation
    trainer.cross_validate(X_train_split, y_train_split, cv=5)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "=" * 60)
    print("✅ DAY 3 TRAINING COMPLETE!")
    print("=" * 60)