"""
PlantPulse: Algorithm 2 - MobileNetV2 Transfer Learning
Leverages ImageNet pre-trained representations with frozen feature base,
custom classification head, and subtle top-layer fine-tuning for maximum generalization.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, Rescaling
# pyrefly: ignore[missing-import]
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataset_utils import get_generators, evaluate_and_save_results, BASE_DIR

def build_mobilenetv2_model(input_shape=(128, 128, 3), num_classes=7):
    """
    Builds a Transfer Learning model using MobileNetV2 pretrained on ImageNet.
    Base model is initially frozen with a custom classification head.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    # Scale from [0, 1] (from generator) to [-1, 1] expected by MobileNetV2
    x = Rescaling(scale=2.0, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model

def train_algo2(initial_epochs=10, fine_tune_epochs=5, batch_size=32):
    print("=" * 60)
    print("ALGORITHM 2: Transfer Learning with MobileNetV2")
    print("=" * 60)
    
    # Load standardized splits
    train_gen, val_gen, test_gen, classes = get_generators(batch_size=batch_size)
    num_classes = len(classes)
    
    model, base_model = build_mobilenetv2_model(input_shape=(128, 128, 3), num_classes=num_classes)
    model.summary()
    
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "algo2.h5")
    
    # -------------------------------------------------------------
    # PHASE 1: Feature Extraction (Frozen Base)
    # -------------------------------------------------------------
    print("\n>>> Phase 1: Training Classification Head (Base Model Frozen) <<<")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_phase1 = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    start_time = time.time()
    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=initial_epochs,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # -------------------------------------------------------------
    # PHASE 2: Light Fine-Tuning (Unfreezing top layers of MobileNetV2)
    # -------------------------------------------------------------
    print("\n>>> Phase 2: Light Fine-Tuning Top Layers of MobileNetV2 <<<")
    base_model.trainable = True
    
    # Freeze all layers except the top 25 layers
    fine_tune_at = len(base_model.layers) - 25
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    print(f"Total base layers: {len(base_model.layers)}, Fine-tuning from layer {fine_tune_at} onwards.")
    
    # Recompile with smaller learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_phase2 = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    ]
    
    history_phase2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tune_epochs,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\n[Algorithm 2] Total training completed in {training_time:.2f} seconds.")
    
    # Merge histories for unified reporting
    combined_history = {}
    for key in history_phase1.history.keys():
        combined_history[key] = history_phase1.history[key] + history_phase2.history.get(key, [])
        
    # Save final model state
    model.save(checkpoint_path)
    print(f"[Algorithm 2] Model saved to: {checkpoint_path}")
    
    # Evaluate on unseen test set and save results
    metrics = evaluate_and_save_results(
        model=model,
        test_gen=test_gen,
        classes=classes,
        history_dict=combined_history,
        training_time=training_time,
        algo_prefix="algo2",
        algo_name="MobileNetV2 Transfer Learning"
    )
    
    print("\n--- Algorithm 2 Test Summary ---")
    print(f"Accuracy:  {metrics['test_accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision_weighted']*100:.2f}%")
    print(f"Recall:    {metrics['recall_weighted']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_weighted']*100:.2f}%")
    print("-" * 35)
    
    return metrics

if __name__ == "__main__":
    train_algo2(initial_epochs=10, fine_tune_epochs=5)
