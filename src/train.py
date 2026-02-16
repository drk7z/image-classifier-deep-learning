"""
Training script for image classification model.

This module handles data loading, augmentation, and model training.
"""

import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)
import matplotlib.pyplot as plt
from datetime import datetime

from .model import create_cnn_model, compile_model


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
    
    def train(self, epochs=50, batch_size=32, learning_rate=0.001, 
              use_augmentation=True, model_name='cnn_classifier'):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_augmentation: Whether to use data augmentation
            model_name: Name for the model
        """
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
        print("Creating model...")
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
        
        # Train model
        print("Starting training...")
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen)
        )
        
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
