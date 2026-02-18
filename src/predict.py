"""
Prediction script for image classification.

This module handles making predictions on new images.
"""

import os
from pathlib import Path
import numpy as np

# Runtime safety defaults for constrained environments (e.g., Streamlit Cloud)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import tensorflow as tf
from PIL import Image


_TF_RUNTIME_CONFIGURED = False


def _configure_tensorflow_runtime():
    global _TF_RUNTIME_CONFIGURED
    if _TF_RUNTIME_CONFIGURED:
        return

    try:
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass

        intra_threads = int(os.getenv('TF_NUM_INTRAOP_THREADS', '1'))
        inter_threads = int(os.getenv('TF_NUM_INTEROP_THREADS', '1'))
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    except Exception:
        pass
    finally:
        _TF_RUNTIME_CONFIGURED = True


class ImageClassifier:
    """Class for making predictions with trained model."""
    
    def __init__(self, model_path, class_names=None):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained model
            class_names: List of class names
        """
        _configure_tensorflow_runtime()
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names or ['cat', 'dog']
        self.img_size = 224
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image array
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize
        img = img.resize((self.img_size, self.img_size))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    def predict(self, image_path, return_confidence=False):
        """
        Make prediction on single image.
        
        Args:
            image_path: Path to image file
            return_confidence: Whether to return confidence scores
        
        Returns:
            Predicted class name and optionally confidence
        """
        img_array, _ = self.preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        if return_confidence:
            return predicted_class, confidence, predictions[0]
        
        return predicted_class
    
    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            List of predictions
        """
        results = []
        for image_path in image_paths:
            pred_class, confidence, _ = self.predict(
                image_path,
                return_confidence=True
            )
            results.append({
                'image': image_path,
                'prediction': pred_class,
                'confidence': float(confidence)
            })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image.
        
        Args:
            image_path: Path to image file
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        img_array, img = self.preprocess_image(image_path)
        
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2%}')
        plt.axis('off')
        
        # Plot confidence scores
        plt.subplot(1, 2, 2)
        colors = ['green' if i == predicted_class_idx else 'gray' 
                 for i in range(len(self.class_names))]
        plt.bar(self.class_names, predictions[0], color=colors)
        plt.ylabel('Confidence')
        plt.title('Prediction Confidence')
        plt.ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == '__main__':
    # Example usage
    classifier = ImageClassifier(
        model_path='models/cnn_classifier_final.h5',
        class_names=['cat', 'dog']
    )
    
    # Single prediction
    image_path = 'path/to/image.jpg'
    pred_class, confidence = classifier.predict(image_path, return_confidence=True)
    print(f"Prediction: {pred_class} ({confidence:.2%})")
    
    # Visualize
    classifier.visualize_prediction(image_path)
