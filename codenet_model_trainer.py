"""
CodeNet Model Trainer

This module provides training capabilities for runtime error prediction models
using the Project CodeNet dataset.
"""

import os
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from codenet_data_loader import CodeNetDataLoader, CodeNetSample
from code_preprocessor import CodePreprocessor


class CodeNetDataset:
    """Dataset wrapper for CodeNet samples"""

    def __init__(self, samples: List[CodeNetSample], preprocessor: CodePreprocessor):
        self.samples = samples
        self.preprocessor = preprocessor
        self.label_encoder = LabelEncoder()

        # Extract labels
        labels = [s.label for s in samples]
        self.labels_encoded = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.samples)

    def get_features_and_labels(self) -> Tuple[List[Dict], List[int]]:
        """Extract features and labels for all samples"""
        features = []
        labels = []

        for i, sample in enumerate(self.samples):
            try:
                # Extract features
                feat = self.preprocessor.extract_features(sample.code, sample.language.lower())
                features.append(feat)
                labels.append(self.labels_encoded[i])
            except Exception as e:
                logging.warning(f"Failed to extract features for {sample.submission_id}: {e}")

        return features, labels


class RuntimeErrorPredictor:
    """
    Machine learning model trainer for runtime error prediction
    using Project CodeNet data
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the model trainer

        Args:
            model_type: Type of model to train
                       ("random_forest", "gradient_boosting", "logistic_regression", "svm")
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.preprocessor = CodePreprocessor()

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for model training. Install with: pip install scikit-learn")

        # Initialize model based on type
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ML model based on model_type"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.logger.info(f"Initialized {self.model_type} model")

    def prepare_data(self, samples: List[CodeNetSample]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from CodeNet samples

        Args:
            samples: List of CodeNetSample objects

        Returns:
            Tuple of (features_array, labels_array, label_names)
        """
        self.logger.info(f"Preparing data from {len(samples)} samples...")

        dataset = CodeNetDataset(samples, self.preprocessor)
        features_list, labels = dataset.get_features_and_labels()

        self.logger.info(f"Extracted features from {len(features_list)} samples")

        # Convert features to numpy array
        if not features_list:
            raise ValueError("No features extracted from samples")

        # Get feature names from first sample
        self.feature_names = list(features_list[0].keys())

        # Convert to numpy array
        X = np.array([[f.get(name, 0) for name in self.feature_names] for f in features_list])
        y = np.array(labels)

        # Store label encoder
        self.label_encoder = dataset.label_encoder

        self.logger.info(f"Feature matrix shape: {X.shape}")
        self.logger.info(f"Labels shape: {y.shape}")
        self.logger.info(f"Label classes: {list(self.label_encoder.classes_)}")

        return X, y, list(self.label_encoder.classes_)

    def train(self, train_samples: List[CodeNetSample],
             val_samples: Optional[List[CodeNetSample]] = None) -> Dict[str, Any]:
        """
        Train the model on CodeNet samples

        Args:
            train_samples: Training samples
            val_samples: Validation samples (optional)

        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Training {self.model_type} model...")

        # Prepare training data
        X_train, y_train, label_names = self.prepare_data(train_samples)

        # Train the model
        self.logger.info("Fitting model...")
        self.model.fit(X_train, y_train)

        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        self.logger.info(f"Training accuracy: {train_acc:.4f}")

        metrics = {
            'train_accuracy': train_acc,
            'num_train_samples': len(train_samples),
            'model_type': self.model_type,
            'label_names': label_names,
        }

        # Evaluate on validation set if provided
        if val_samples:
            val_metrics = self.evaluate(val_samples)
            metrics['val_accuracy'] = val_metrics['accuracy']
            metrics['val_report'] = val_metrics['classification_report']

        return metrics

    def evaluate(self, test_samples: List[CodeNetSample]) -> Dict[str, Any]:
        """
        Evaluate the model on test samples

        Args:
            test_samples: Test samples

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        self.logger.info(f"Evaluating on {len(test_samples)} samples...")

        # Prepare test data
        dataset = CodeNetDataset(test_samples, self.preprocessor)
        features_list, y_true = dataset.get_features_and_labels()

        X_test = np.array([[f.get(name, 0) for name in self.feature_names] for f in features_list])

        # Predict
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        self.logger.info(f"Test accuracy: {accuracy:.4f}")
        self.logger.info(f"Test F1 score: {f1:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'num_samples': len(test_samples)
        }

    def predict(self, code: str, language: str) -> Tuple[str, float]:
        """
        Predict error type for a code sample

        Args:
            code: Source code
            language: Programming language

        Returns:
            Tuple of (predicted_label, confidence)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Extract features
        features = self.preprocessor.extract_features(code, language)
        X = np.array([[features.get(name, 0) for name in self.feature_names]])

        # Predict
        pred = self.model.predict(X)[0]
        label = self.label_encoder.inverse_transform([pred])[0]

        # Get confidence
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 0.8  # Default confidence

        return label, confidence

    def save_model(self, output_path: str):
        """
        Save the trained model to disk

        Args:
            output_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Model saved to {output_path}")

    def load_model(self, model_path: str):
        """
        Load a trained model from disk

        Args:
            model_path: Path to the saved model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data.get('model_type', 'unknown')

        self.logger.info(f"Model loaded from {model_path}")


def train_model_from_codenet(
    codenet_path: Optional[str] = None,
    output_model_path: str = "models/codenet_runtime_error_model.pkl",
    model_type: str = "random_forest",
    max_problems: int = 100,
    languages: Optional[List[str]] = None,
    limit_per_problem: int = 50
) -> RuntimeErrorPredictor:
    """
    Train a model using Project CodeNet data

    Args:
        codenet_path: Path to CodeNet dataset
        output_model_path: Where to save trained model
        model_type: Type of model to train
        max_problems: Maximum number of problems to use
        languages: Languages to include (None for all)
        limit_per_problem: Max samples per problem

    Returns:
        Trained RuntimeErrorPredictor
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Initializing CodeNet data loader...")
    loader = CodeNetDataLoader(codenet_path)

    if not loader.validate_dataset():
        raise ValueError(f"Invalid CodeNet dataset at {loader.codenet_path}")

    logger.info("Loading dataset...")
    samples = loader.load_dataset(
        languages=languages,
        statuses=["Runtime Error", "Accepted", "Time Limit Exceeded", "Memory Limit Exceeded"],
        limit_per_problem=limit_per_problem,
        max_problems=max_problems
    )

    if len(samples) == 0:
        raise ValueError("No samples loaded from dataset")

    logger.info(f"Loaded {len(samples)} samples")

    # Show dataset statistics
    stats = loader.get_dataset_statistics(samples)
    logger.info(f"Dataset statistics: {stats}")

    # Split data
    train_samples, val_samples, test_samples = loader.create_training_split(samples)

    # Initialize and train model
    logger.info(f"Initializing {model_type} model...")
    predictor = RuntimeErrorPredictor(model_type=model_type)

    logger.info("Training model...")
    train_metrics = predictor.train(train_samples, val_samples)
    logger.info(f"Training metrics: {train_metrics}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = predictor.evaluate(test_samples)
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"\n{test_metrics['classification_report']}")

    # Save model
    predictor.save_model(output_model_path)

    # Save metrics
    metrics_path = Path(output_model_path).parent / "training_metrics.json"
    all_metrics = {
        'train_metrics': {k: v for k, v in train_metrics.items() if not isinstance(v, (np.ndarray, list))},
        'test_metrics': {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'},
        'dataset_stats': stats,
        'timestamp': datetime.now().isoformat()
    }

    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")

    return predictor


def main():
    """Example usage of the model trainer"""
    import argparse

    parser = argparse.ArgumentParser(description="Train runtime error prediction model on CodeNet data")
    parser.add_argument("--codenet-path", type=str, help="Path to Project CodeNet dataset")
    parser.add_argument("--output", type=str, default="models/codenet_runtime_error_model.pkl",
                       help="Output path for trained model")
    parser.add_argument("--model-type", type=str, default="random_forest",
                       choices=["random_forest", "gradient_boosting", "logistic_regression", "svm"],
                       help="Type of model to train")
    parser.add_argument("--max-problems", type=int, default=100,
                       help="Maximum number of problems to use")
    parser.add_argument("--limit-per-problem", type=int, default=50,
                       help="Maximum samples per problem")
    parser.add_argument("--languages", type=str, nargs="+",
                       help="Languages to include (e.g., Python C++)")

    args = parser.parse_args()

    try:
        predictor = train_model_from_codenet(
            codenet_path=args.codenet_path,
            output_model_path=args.output,
            model_type=args.model_type,
            max_problems=args.max_problems,
            languages=args.languages,
            limit_per_problem=args.limit_per_problem
        )
        print(f"\nModel training complete! Model saved to {args.output}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
