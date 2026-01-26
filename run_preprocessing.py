import sys
sys.path.insert(0, 'src')

from prepare_dataset import DatasetPreparer

if __name__ == "__main__":
    print("=" * 60)
    print("🎤 A-I DUO VOICE DETECTION - PREPROCESSING")
    print("=" * 60)
    
    preparer = DatasetPreparer()
    X_train, y_train, scaler = preparer.prepare_training_data()
    
    if X_train is not None:
        print("\n✅ PREPROCESSING COMPLETE!")
        print(f"Ready for model training...")
    else:
        print("\n❌ PREPROCESSING FAILED!")
        print("Check data/train/ folder structure")
