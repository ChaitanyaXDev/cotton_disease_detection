"""
PlantPulse: Algorithm 1 - Custom Convolutional Neural Network (From Scratch)
3-block CNN architecture (Conv2D 32 -> 64 -> 128 + MaxPool + Dense 128 + Dropout)
trained end-to-end on cotton foliar pathology datasets.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataset_utils import get_generators, evaluate_and_save_results, BASE_DIR

def build_custom_cnn(input_shape=(128, 128, 3), num_classes=7):
    """
    Builds a robust Custom CNN architecture for cotton disease classification.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Block 1
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Classification Head
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_algo1(epochs=10, batch_size=32):
    print("=" * 60)
    print("ALGORITHM 1: Custom CNN from Scratch")
    print("=" * 60)
    
    # Load standardized splits
    train_gen, val_gen, test_gen, classes = get_generators(batch_size=batch_size)
    num_classes = len(classes)
    
    model = build_custom_cnn(input_shape=(128, 128, 3), num_classes=num_classes)
    model.summary()
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "algo1.h5")
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"\n[Algorithm 1] Training completed in {training_time:.2f} seconds.")
    
    # Save final model state
    model.save(checkpoint_path)
    print(f"[Algorithm 1] Model saved to: {checkpoint_path}")
    
    # Evaluate on unseen test set and save results
    metrics = evaluate_and_save_results(
        model=model,
        test_gen=test_gen,
        classes=classes,
        history_dict=history.history,
        training_time=training_time,
        algo_prefix="algo1",
        algo_name="Sequential 3-Block CNN"
    )
    
    print("\n--- Algorithm 1 Test Summary ---")
    print(f"Accuracy:  {metrics['test_accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision_weighted']*100:.2f}%")
    print(f"Recall:    {metrics['recall_weighted']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_weighted']*100:.2f}%")
    print("-" * 35)
    
    return metrics

if __name__ == "__main__":
    train_algo1(epochs=10)
