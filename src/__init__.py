"""
Image Classifier Deep Learning Package

A complete image classification project using CNNs and Transfer Learning with TensorFlow/Keras.
"""

__version__ = "1.0.0"
__author__ = "Leandro Vieira"
__email__ = "seu.email@example.com"

from .src.model import create_cnn_model, create_transfer_learning_model, compile_model
from .src.train import ImageClassifierTrainer
from .src.evaluate import ModelEvaluator
from .src.predict import ImageClassifier

__all__ = [
    'create_cnn_model',
    'create_transfer_learning_model',
    'compile_model',
    'ImageClassifierTrainer',
    'ModelEvaluator',
    'ImageClassifier'
]
