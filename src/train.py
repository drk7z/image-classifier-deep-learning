"""
Training script for image classification model.

This module handles data loading, augmentation, and model training.
"""

import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)
import matplotlib.pyplot as plt
from datetime import datetime

from .model import create_cnn_model, create_transfer_learning_model, compile_model


class ImageClassifierTrainer:
    """Trainer class for image classification models."""
    
    def __init__(self, data_dir, model_dir='models', log_dir='logs'):
        """
        Initialize trainer.
        
        Args:
            data_dir: Path to data directory
            model_dir: Directory to save models
            log_dir: Directory for TensorBoard logs
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories if they don't exist
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        self.history = None
        self.model = None
    
    def create_data_generators(self, img_size=224, batch_size=32, augment=True):
        """
        Create data generators for training and validation.
        
        Args:
            img_size: Image size (square)
            batch_size: Batch size for training
            augment: Whether to apply data augmentation
        
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        # Training data augmentation
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Validation data (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            str(self.data_dir / 'train'),
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Load validation data
        validation_generator = val_datagen.flow_from_directory(
            str(self.data_dir / 'validation'),
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, validation_generator

    def configure_hardware_acceleration(self, enable_xla=True, enable_mixed_precision=True):
        """Configure TensorFlow hardware acceleration settings."""
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')

        print("\n=== Hardware Info ===")
        print(f"CPUs available: {len(cpus)}")
        print(f"GPUs available: {len(gpus)}")

        if enable_xla:
            tf.config.optimizer.set_jit(True)
            print("XLA JIT: enabled")

        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass

            if enable_mixed_precision:
                mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision: enabled (mixed_float16)")
        else:
            print("GPU not detected. Training will use CPU.")

    def _merge_histories(self, base_history, extra_history):
        """Merge Keras History objects from multi-stage training."""
        if base_history is None:
            return extra_history
        if extra_history is None:
            return base_history

        for key, values in extra_history.history.items():
            if key in base_history.history:
                base_history.history[key].extend(values)
            else:
                base_history.history[key] = values

        return base_history
    
    def train(self, epochs=50, batch_size=32, learning_rate=0.001,
              use_augmentation=True, model_name=None, model_type='cnn',
              fine_tune=True, fine_tune_layers=30, fine_tune_epochs=5):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_augmentation: Whether to use data augmentation
            model_name: Name for the model. If None, inferred from model_type.
            model_type: 'cnn' or 'transfer'
            fine_tune: Whether to run a fine-tuning stage (transfer model only)
            fine_tune_layers: Number of base model layers to unfreeze in fine-tuning
            fine_tune_epochs: Number of epochs for fine-tuning stage
        """
        model_type = model_type.lower().strip()
        if model_type not in ('cnn', 'transfer'):
            raise ValueError("model_type must be 'cnn' or 'transfer'")

        if model_name is None:
            model_name = 'cnn_classifier' if model_type == 'cnn' else 'transfer_learning'

        self.configure_hardware_acceleration()

        print("Creating data generators...")
        train_gen, val_gen = self.create_data_generators(
            batch_size=batch_size,
            augment=use_augmentation
        )
        
        # Get number of classes
        num_classes = train_gen.num_classes
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {train_gen.class_indices}")
        
        # Create and compile model
        print(f"Creating {model_type} model...")
        if model_type == 'transfer':
            self.model = create_transfer_learning_model(num_classes=num_classes)
        else:
            self.model = create_cnn_model(num_classes=num_classes)

        self.model = compile_model(self.model, learning_rate=learning_rate)
        
        # Print model summary
        self.model.summary()
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"{model_name}_{timestamp}.h5"
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(model_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(self.log_dir / timestamp),
                histogram_freq=1
            )
        ]
        
        # Train model - stage 1
        total_epochs = max(int(epochs), 1)
        effective_fine_tune = (
            model_type == 'transfer' and
            fine_tune and
            int(fine_tune_epochs) > 0 and
            total_epochs > 1
        )

        stage1_epochs = total_epochs
        stage2_epochs = 0
        if effective_fine_tune:
            stage2_epochs = min(int(fine_tune_epochs), total_epochs - 1)
            stage1_epochs = total_epochs - stage2_epochs

        print(f"Starting training (stage 1 / {stage1_epochs} epochs)...")
        self.history = self.model.fit(
            train_gen,
            epochs=stage1_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen)
        )

        # Fine-tuning stage for transfer learning
        if effective_fine_tune and stage2_epochs > 0:
            print(f"Starting fine-tuning (stage 2 / {stage2_epochs} epochs)...")

            base_model = self.model.layers[0]
            base_model.trainable = True

            if fine_tune_layers > 0:
                for layer in base_model.layers[:-fine_tune_layers]:
                    layer.trainable = False

            self.model = compile_model(self.model, learning_rate=learning_rate * 0.1)

            fine_tune_history = self.model.fit(
                train_gen,
                epochs=stage1_epochs + stage2_epochs,
                initial_epoch=stage1_epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                steps_per_epoch=len(train_gen),
                validation_steps=len(val_gen)
            )
            self.history = self._merge_histories(self.history, fine_tune_history)
        
        # Save model and history
        final_model_path = self.model_dir / f"{model_name}_final_{timestamp}.h5"
        self.model.save(str(final_model_path))
        
        history_path = self.model_dir / f"{model_name}_history_{timestamp}.json"
        self._save_history(str(history_path))
        
        print(f"Model saved to {final_model_path}")
        print(f"History saved to {history_path}")
        
        return self.history
    
    def _save_history(self, path):
        """Save training history to JSON."""
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=4)
    
    def plot_history(self, show=True):
        """Plot training history."""
        if self.history is None:
            print("No history to plot. Train model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig


if __name__ == '__main__':
    # Example usage
    trainer = ImageClassifierTrainer(data_dir='data')
    history = trainer.train(epochs=50, batch_size=32)
    trainer.plot_history()
