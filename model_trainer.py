"""
Model Trainer for Error Detection

This module provides functionality to train ML models for error detection
using the extracted features from CodeNet data.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from code_preprocessor import CodePreprocessor
from error_predictor import ErrorType


class SimpleNeuralNet(nn.Module):
    """Simple neural network for error classification"""

    def __init__(self, input_size: int, hidden_size: int = 128, num_classes: int = 8):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ModelTrainer:
    """
    Handles training of ML models for error detection
    """

    def __init__(self, data_dir: str = "data/codenet_extract"):
        self.data_dir = Path(data_dir)
        self.preprocessor = CodePreprocessor()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def load_training_data(self) -> Tuple[List[Dict], List[str]]:
        """
        Load training data from JSONL files

        Returns:
            Tuple of (features, labels) lists
        """
        features = []
        labels = []

        # Load train, validation, and test data
        for split in ['train', 'valid', 'test']:
            jsonl_file = self.data_dir / f"{split}.jsonl"

            if not jsonl_file.exists():
                self.logger.warning(f"Data file not found: {jsonl_file}")
                continue

            self.logger.info(f"Loading {split} data from {jsonl_file}")

            with jsonl_file.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())

                        # Extract features from code
                        code = record['code']
                        language = record['language']
                        label = record['label']

                        # Skip if label is unknown or empty
                        if not label or label == 'unknown':
                            continue

                        code_features = self.preprocessor.extract_features(code, language)
                        features.append(code_features)
                        labels.append(label)

                        if line_num % 1000 == 0:
                            self.logger.info(f"  Processed {line_num} records from {split}")

                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.warning(f"Error processing line {line_num} in {split}: {e}")

        self.logger.info(f"Loaded {len(features)} training samples")
        return features, labels

    def prepare_features(self, feature_dicts: List[Dict]) -> np.ndarray:
        """
        Convert feature dictionaries to numpy array

        Args:
            feature_dicts: List of feature dictionaries

        Returns:
            NumPy array of features
        """
        if not feature_dicts:
            return np.array([])

        # Get all possible feature names
        all_features = set()
        for feat_dict in feature_dicts:
            all_features.update(feat_dict.keys())

        feature_names = sorted(all_features)
        self.feature_names = feature_names

        # Convert to matrix
        feature_matrix = []
        for feat_dict in feature_dicts:
            row = [feat_dict.get(name, 0.0) for name in feature_names]
            feature_matrix.append(row)

        return np.array(feature_matrix, dtype=np.float32)

    def train_sklearn_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model_type: str = 'random_forest'
    ) -> Any:
        """
        Train a scikit-learn model

        Args:
            features: Feature matrix
            labels: Encoded labels
            model_type: Type of model ('random_forest', 'logistic', 'svm')

        Returns:
            Trained model
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for training sklearn models")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic':
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        self.logger.info(f"Training {model_type} model...")
        model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        self.logger.info(f"Training accuracy: {train_score:.3f}")
        self.logger.info(f"Test accuracy: {test_score:.3f}")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        self.logger.info(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Detailed evaluation
        y_pred = model.predict(X_test_scaled)
        self.logger.info("\nClassification Report:")
        self.logger.info(classification_report(y_test, y_pred))

        return model

    def train_neural_network(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> nn.Module:
        """
        Train a PyTorch neural network

        Args:
            features: Feature matrix
            labels: Encoded labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer

        Returns:
            Trained neural network model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training neural networks")

        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels)

        # Split data
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_size = features.shape[1]
        num_classes = len(np.unique(labels))
        model = SimpleNeuralNet(input_size, 128, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        self.logger.info("Training neural network...")
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            if (epoch + 1) % 10 == 0:
                accuracy = 100 * correct / total
                avg_loss = total_loss / len(train_loader)
                self.logger.info(f"Epoch [{epoch+1}/{epochs}], "
                               f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        test_accuracy = 100 * correct / total
        self.logger.info(f"Test Accuracy: {test_accuracy:.2f}%")

        return model

    def save_model(self, model: Any, model_path: str, model_type: str = 'sklearn') -> None:
        """
        Save trained model to disk

        Args:
            model: Trained model
            model_path: Path to save the model
            model_type: Type of model ('sklearn' or 'pytorch')
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if model_type == 'sklearn':
            # Save sklearn model with preprocessing
            model_data = {
                'model': model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': getattr(self, 'feature_names', [])
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

        elif model_type == 'pytorch':
            # Save PyTorch model
            torch.save(model.state_dict(), model_path)

            # Also save model metadata
            metadata = {
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': getattr(self, 'feature_names', [])
            }

            metadata_path = model_path.with_suffix('.metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

        self.logger.info(f"Model saved to {model_path}")

    def train_and_save(
        self,
        model_type: str = 'random_forest',
        output_path: str = 'models/error_detector.pkl'
    ) -> None:
        """
        Complete training pipeline

        Args:
            model_type: Type of model to train
            output_path: Path to save the trained model
        """
        # Load data
        feature_dicts, labels = self.load_training_data()

        if not feature_dicts:
            raise ValueError("No training data found")

        # Prepare features
        features = self.prepare_features(feature_dicts)

        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        self.logger.info(f"Training data shape: {features.shape}")
        self.logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        self.logger.info(f"Classes: {self.label_encoder.classes_}")

        # Train model
        if model_type in ['random_forest', 'logistic', 'svm']:
            model = self.train_sklearn_model(features, encoded_labels, model_type)
            self.save_model(model, output_path, 'sklearn')

        elif model_type == 'neural_network':
            model = self.train_neural_network(features, encoded_labels)
            self.save_model(model, output_path, 'pytorch')

        else:
            raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Train error detection model")
    parser.add_argument('--data-dir', default='data/codenet_extract',
                       help='Directory containing training data')
    parser.add_argument('--model-type', default='random_forest',
                       choices=['random_forest', 'logistic', 'svm', 'neural_network'],
                       help='Type of model to train')
    parser.add_argument('--output', default='models/error_detector.pkl',
                       help='Output path for trained model')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        trainer = ModelTrainer(args.data_dir)
        trainer.train_and_save(args.model_type, args.output)
        print(f"Model training complete. Model saved to {args.output}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()