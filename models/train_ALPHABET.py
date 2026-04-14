import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONFIGURATION ---
DATA_PATH = r"C:\Users\Harshit Sharma\OneDrive\Desktop\B.Tech Project\data\processed\master_features.csv"
MODEL_DIR = r"C:\Users\Harshit Sharma\OneDrive\Desktop\B.Tech Project\models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_system():
    # 1. Load Data
    print("📂 Loading master_features.csv...")
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Identify feature groups
    raw_cols = [c for c in X.columns if c.startswith('r_')]
    agd_cols = [c for c in X.columns if c.startswith('d_') or c.startswith('a_')]
    
    # 2. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    def run_experiment(name, features):
        print(f"\n🚀 Training: {name}...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train[features], y_train)
        
        preds = clf.predict(X_test[features])
        acc = accuracy_score(y_test, preds)
        
        print(f"✅ Accuracy: {acc*100:.2f}%")
        return clf, acc, preds

    # 3. The Experiments
    model_raw, acc_raw, _ = run_experiment("Baseline (MediaPipe Only)", raw_cols)
    model_agd, acc_agd, _ = run_experiment("Research (AGD Only)", agd_cols)
    model_hybrid, acc_hybrid, hybrid_preds = run_experiment("Hybrid (Raw + AGD)", raw_cols + agd_cols)

    # 4. Save the Hybrid Model (The Final Product)
    model_path = os.path.join(MODEL_DIR, "sign_language_model.pkl")
    joblib.dump(model_hybrid, model_path)
    print(f"\n💾 Final model saved at: {model_path}")

    # 5. Save Metrics & Plots
    # A: Accuracy Comparison Plot
    plt.figure(figsize=(10, 6))
    labels = ['Baseline (63)', 'AGD (7)', 'Hybrid (70)']
    accs = [acc_raw, acc_agd, acc_hybrid]
    colors = ['#ff7675', '#74b9ff', '#55efc4']
    
    bars = plt.bar(labels, accs, color=colors)
    plt.title('Feature Set Comparison: Accuracy Metrics', fontsize=14)
    plt.ylabel('Score (0.0 - 1.0)')
    plt.ylim(0, 1.1)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval*100:.1f}%', ha='center', fontweight='bold')
    
    plt.savefig(os.path.join(MODEL_DIR, "accuracy_comparison.png"))
    
    # B: Confusion Matrix for the Hybrid Model
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, hybrid_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title('Confusion Matrix: Hybrid System')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))

    # C: Save Numerical Report
    report = classification_report(y_test, hybrid_preds, output_dict=True)
    with open(os.path.join(MODEL_DIR, "performance_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print("\n📊 ALL METRICS SAVED:")
    print(f"- accuracy_comparison.png")
    print(f"- confusion_matrix.png")
    print(f"- performance_report.json")

if __name__ == "__main__":
    train_system()