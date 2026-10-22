import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
# pyrefly: ignore[missing-import]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
PlantPulse: Dataset Utilities Module
Handles reproducible stratified 70/15/15 dataset splitting, data augmentation pipelines,
and metric calculation with confusion matrix and training curve plotting.
"""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
RANDOM_SEED = 42

def load_data_splits(dataset_dir=DATASET_DIR, seed=RANDOM_SEED):
    """
    Scans the dataset directory and produces reproducible, stratified
    train (70%), validation (15%), and test (15%) dataframes.
    """
    filepaths = []
    labels = []
    
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(cls)
                
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    
    # 70% Train, 30% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df['label']
    )
    
    # Split temp 50-50 into 15% Validation and 15% Test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df['label']
    )
    
    return train_df, val_df, test_df, classes

def get_generators(batch_size=BATCH_SIZE, target_size=IMG_SIZE, seed=RANDOM_SEED):
    """
    Returns train, validation, and test generators with identical splits.
    Train generator applies data augmentation (rotation, zoom, flips).
    Validation and test generators only rescale images.
    """
    train_df, val_df, test_df, classes = load_data_splits(seed=seed)
    
    # Data Augmentation for training (mild, natural leaf transformations)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescale for validation and test
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=True,
        seed=seed
    )
    
    val_gen = eval_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=False
    )
    
    test_gen = eval_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen, classes

def evaluate_and_save_results(model, test_gen, classes, history_dict, training_time, algo_prefix, algo_name):
    """
    Evaluates model on test_gen, computes metrics, saves confusion matrix,
    training curves, and writes metrics JSON.
    """
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Predict on test set
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes
    
    # Calculate performance metrics
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # Detailed per-class report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    
    metrics = {
        'algorithm_name': algo_name,
        'algorithm_id': algo_prefix,
        'test_accuracy': acc,
        'precision_macro': float(p_macro),
        'recall_macro': float(r_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(p_weighted),
        'recall_weighted': float(r_weighted),
        'f1_weighted': float(f1_weighted),
        'training_time_seconds': float(training_time),
        'classes': classes,
        'classification_report': report,
        'history': {k: [float(v) for v in vals] for k, vals in history_dict.items()}
    }
    
    # Save metrics JSON
    metrics_path = os.path.join(results_dir, f"{algo_prefix}_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"[{algo_name}] Metrics saved to: {metrics_path}")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"{algo_name} - Confusion Matrix\nAccuracy: {acc*100:.2f}%", fontsize=14, pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual Class", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, f"{algo_prefix}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[{algo_name}] Confusion matrix saved to: {cm_path}")
    
    # Accuracy and Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history_dict.get('accuracy', [])) + 1)
    
    # Accuracy Curve
    axes[0].plot(epochs_range, history_dict.get('accuracy', []), 'o-', label='Train Accuracy', color='#2563eb')
    axes[0].plot(epochs_range, history_dict.get('val_accuracy', []), 's--', label='Val Accuracy', color='#16a34a')
    axes[0].set_title(f"{algo_name} - Accuracy Curve", fontsize=13)
    axes[0].set_xlabel("Epochs", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    # Loss Curve
    axes[1].plot(epochs_range, history_dict.get('loss', []), 'o-', label='Train Loss', color='#dc2626')
    axes[1].plot(epochs_range, history_dict.get('val_loss', []), 's--', label='Val Loss', color='#d97706')
    axes[1].set_title(f"{algo_name} - Loss Curve", fontsize=13)
    axes[1].set_xlabel("Epochs", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    curves_path = os.path.join(results_dir, f"{algo_prefix}_training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"[{algo_name}] Curves saved to: {curves_path}")
    
    return metrics
