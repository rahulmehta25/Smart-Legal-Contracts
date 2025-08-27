"""
End-to-End ML Pipeline Test

Test the complete ML pipeline with a simplified model setup
"""

import os
import sys
sys.path.append('.')

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from app.ml.features import LegalFeatureExtractor
from data.training_data import TrainingDataGenerator

def test_complete_pipeline():
    """Test the complete ML pipeline end-to-end"""
    
    print("=" * 60)
    print("TESTING COMPLETE ML PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate training data
    print("\n1. Generating training data...")
    generator = TrainingDataGenerator()
    df = generator.create_training_dataset(
        num_synthetic_positive=200,
        num_synthetic_negative=200,
        num_ambiguous=50,
        include_variations=False
    )
    
    texts = df['text'].tolist()
    labels = df['label'].values
    
    print(f"Generated {len(texts)} examples")
    print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    # Step 2: Create feature extractor
    print("\n2. Creating and fitting feature extractor...")
    feature_extractor = LegalFeatureExtractor(
        max_features_tfidf=1000,
        ngram_range=(1, 2),
        use_legal_keywords=True,
        use_structure_features=True,
        use_statistical_features=True
    )
    
    feature_extractor.fit(texts)
    X = feature_extractor.transform(texts)
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Step 3: Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {X_train.shape[0]} examples")
    print(f"Test set: {X_test.shape[0]} examples")
    
    # Step 4: Train a simple model
    print("\n4. Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Step 5: Evaluate
    print("\n5. Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Step 6: Save models
    print("\n6. Saving models...")
    os.makedirs('models', exist_ok=True)
    
    # Save feature extractor
    feature_extractor.save('models/feature_extractor.pkl')
    
    # Save model
    joblib.dump(model, 'models/random_forest_model.pkl')
    
    print("Models saved successfully!")
    
    # Step 7: Test prediction on new examples
    print("\n7. Testing prediction on new examples...")
    
    test_texts = [
        "Any dispute arising out of these terms shall be resolved through binding arbitration.",
        "You may bring legal action in any court of competent jurisdiction.",
        "All disputes will be settled by arbitration administered by AAA.",
        "This privacy policy explains our data collection practices."
    ]
    
    expected_labels = [1, 0, 1, 0]
    
    # Transform test texts
    X_new = feature_extractor.transform(test_texts)
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]
    
    print("\nPrediction Results:")
    print("-" * 40)
    
    correct = 0
    for i, (text, pred, prob, expected) in enumerate(zip(test_texts, predictions, probabilities, expected_labels)):
        is_correct = pred == expected
        correct += is_correct
        
        print(f"\nText {i+1}: {text[:50]}...")
        print(f"Prediction: {'ARBITRATION' if pred else 'NO ARBITRATION'}")
        print(f"Confidence: {prob:.3f}")
        print(f"Expected: {'ARBITRATION' if expected else 'NO ARBITRATION'}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
    
    accuracy = correct / len(test_texts)
    print(f"\nTest Accuracy: {accuracy:.3f} ({correct}/{len(test_texts)})")
    
    # Step 8: Test feature importance
    print("\n8. Analyzing feature importance...")
    feature_names = feature_extractor.get_feature_names()
    importances = model.feature_importances_
    
    # Get top 10 features
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most important features:")
    for i, (name, importance) in enumerate(feature_importance[:10]):
        print(f"{i+1:2d}. {name[:50]:<50} {importance:.4f}")
    
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'training_accuracy': accuracy,
        'training_f1': f1,
        'test_accuracy': accuracy,
        'feature_count': X.shape[1],
        'model_saved': True
    }

if __name__ == "__main__":
    results = test_complete_pipeline()