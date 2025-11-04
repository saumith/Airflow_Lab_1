import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
import base64

def load_data():
    print("Loading data for SVM...")
    csv_path = os.path.join(os.path.dirname(__file__), "../data/file.csv")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} samples")
    
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    print("Preprocessing data...")
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    
    df = df.dropna()
    
    target_column = 'target'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"  - Features: {X.shape}")
    print(f"  - Classes: {sorted(y.unique())}")
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    preprocessed_data = (X_scaled, y.values)
    
    model_dir = os.path.join(os.path.dirname(__file__), "../model")
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    serialized_data = pickle.dumps(preprocessed_data)
    return base64.b64encode(serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    print("\nTraining SVM model...")
    data_bytes = base64.b64decode(data_b64)
    X, y = pickle.loads(data_bytes)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  - Train: {len(X_train)} samples")
    print(f"  - Test: {len(X_test)} samples")
    
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    print("  - Model trained")
    
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"MODEL METRICS")
    print(f"{'='*50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"{'='*50}\n")
    
    model_dir = os.path.join(os.path.dirname(__file__), "../model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    with open(model_path, "wb") as f:
        pickle.dump(svm_model, f)
    print(f"  - Model saved")
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    return metrics


def load_model_elbow(filename: str, metrics: dict):
    print("\nLoading model and making predictions...")
    
    model_dir = os.path.join(os.path.dirname(__file__), "../model")
    model_path = os.path.join(model_dir, filename)
    svm_model = pickle.load(open(model_path, "rb"))
    
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    scaler = pickle.load(open(scaler_path, "rb"))
    
    csv_test_path = os.path.join(os.path.dirname(__file__), "../data/test.csv")
    df_test = pd.read_csv(csv_test_path)
    df_test = df_test.dropna()
    
    if 'target' in df_test.columns:
        df_test = df_test.drop(columns=['target'])
    
    X_test_scaled = scaler.transform(df_test)
    predictions = svm_model.predict(X_test_scaled)
    
    print(f"\n{'='*50}")
    print(f"PREDICTIONS")
    print(f"{'='*50}")
    print(f"  Total: {len(predictions)} samples")
    print(f"  First 10: {predictions[:10]}")
    print(f"  Classes: {sorted(set(predictions))}")
    print(f"  Distribution:")
    
    # Correct way: only one loop
    unique_classes, class_counts = np.unique(predictions, return_counts=True)
    for label, count in zip(unique_classes, class_counts):
        print(f"    Class {label}: {count} samples")
    
    print(f"\n  Accuracy: {metrics['accuracy']:.4f}")
    print(f"{'='*50}\n")
    
    result = {
        "predictions": predictions.tolist(),
        "metrics": metrics,
        "num_predictions": len(predictions),
        "unique_classes": sorted(set(predictions.tolist()))
    }
    return result

# Run pipeline
data = load_data()
print("✓ Data loaded")

preprocessed = data_preprocessing(data)
print("✓ Data preprocessed")

metrics = build_save_model(preprocessed, 'svm_model.pkl')
print("✓ Model built and saved")

result = load_model_elbow('svm_model.pkl', metrics)
print("✓ Predictions made")
print(result)