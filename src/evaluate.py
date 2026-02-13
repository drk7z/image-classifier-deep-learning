"""
Evaluation script for image classification model.

This module handles model evaluation and metric computation.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Class for evaluating image classification models."""
    
    def __init__(self, model_path, data_dir='data'):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            data_dir: Path to data directory
        """
        self.model = tf.keras.models.load_model(model_path)
        self.data_dir = Path(data_dir)
        self.predictions = None
        self.true_labels = None
        self.class_names = None
    
    def evaluate_on_test_set(self, img_size=224, batch_size=32):
        """
        Evaluate model on test set.
        
        Args:
            img_size: Image size (square)
            batch_size: Batch size
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Create test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            str(self.data_dir / 'test'),
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.class_names = list(test_generator.class_indices.keys())
        
        # Evaluate
        print("Evaluating on test set...")
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            test_generator,
            steps=len(test_generator)
        )
        
        # Get predictions
        self.predictions = self.model.predict(test_generator)
        self.true_labels = test_generator.classes
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
        
        return metrics
    
    def get_confusion_matrix(self):
        """
        Compute confusion matrix.
        
        Returns:
            Confusion matrix
        """
        if self.predictions is None:
            raise ValueError("Run evaluate_on_test_set first")
        
        pred_labels = np.argmax(self.predictions, axis=1)
        cm = confusion_matrix(self.true_labels, pred_labels)
        
        return cm
    
    def get_classification_report(self):
        """
        Get detailed classification report.
        
        Returns:
            Classification report string
        """
        if self.predictions is None:
            raise ValueError("Run evaluate_on_test_set first")
        
        pred_labels = np.argmax(self.predictions, axis=1)
        report = classification_report(
            self.true_labels,
            pred_labels,
            target_names=self.class_names
        )
        
        return report
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Optional path to save figure
        """
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, save_path=None):
        """
        Plot ROC curves for each class.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.predictions is None:
            raise ValueError("Run evaluate_on_test_set first")
        
        # One-hot encode true labels
        from sklearn.preprocessing import label_binarize
        true_labels_onehot = label_binarize(
            self.true_labels,
            classes=range(len(self.class_names))
        )
        
        plt.figure(figsize=(10, 8))
        
        for i in range(len(self.class_names)):
            fpr, tpr, _ = roc_curve(true_labels_onehot[:, i], self.predictions[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == '__main__':
    # Example usage
    evaluator = ModelEvaluator(model_path='models/cnn_classifier_final.h5')
    metrics = evaluator.evaluate_on_test_set()
    
    print("\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\n=== Classification Report ===")
    print(evaluator.get_classification_report())
    
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curves()
