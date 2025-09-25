"""
Error Detection Model - Main Prediction Interface

This module provides the main interface for predicting potential errors in code files
using static analysis combined with machine learning predictions.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from code_preprocessor import CodePreprocessor
from error_reporter import ErrorReporter


class ErrorType(Enum):
    """Types of errors the model can predict"""
    NO_ERROR = "no_error"
    RUNTIME_ERROR = "runtime_error"
    COMPILE_ERROR = "compile_error"
    LOGIC_ERROR = "logic_error"
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class PredictionResult:
    """Result of error prediction for a code file"""
    file_path: str
    language: str
    error_type: ErrorType
    confidence: float
    line_number: Optional[int] = None
    error_message: Optional[str] = None
    suggestions: Optional[List[str]] = None


class ErrorDetectionModel:
    """
    Main error detection model that combines static analysis with ML predictions
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.preprocessor = CodePreprocessor()
        self.reporter = ErrorReporter()
        self.logger = self._setup_logging()

        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the error detection system"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained error detection model"""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        try:
            # Try loading different model formats
            if model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info(f"Loaded pickle model from {model_path}")

            elif TORCH_AVAILABLE and model_path.suffix in ['.pt', '.pth']:
                self.model = torch.load(model_path, map_location='cpu')
                self.logger.info(f"Loaded PyTorch model from {model_path}")

            elif model_path.is_dir() and TRANSFORMERS_AVAILABLE:
                # Load transformer model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModel.from_pretrained(model_path)
                self.logger.info(f"Loaded transformer model from {model_path}")

            else:
                raise ValueError(f"Unsupported model format: {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict_file(self, file_path: str) -> PredictionResult:
        """
        Predict potential errors in a single code file

        Args:
            file_path: Path to the code file to analyze

        Returns:
            PredictionResult with error predictions and confidence scores
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read and preprocess the code
        try:
            code = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            code = file_path.read_text(encoding='latin-1')

        # Detect programming language
        language = self.preprocessor.detect_language(file_path)

        # Perform static analysis
        static_errors = self.preprocessor.static_analysis(code, language)

        # If we have obvious static errors, return immediately
        if static_errors:
            error_type_str = static_errors[0]['type']
            if isinstance(error_type_str, str):
                # Try to convert string to ErrorType
                try:
                    error_type = ErrorType(error_type_str)
                except ValueError:
                    error_type = ErrorType.UNKNOWN
            else:
                error_type = error_type_str

            return PredictionResult(
                file_path=str(file_path),
                language=language,
                error_type=error_type,
                confidence=0.95,
                line_number=static_errors[0].get('line'),
                error_message=static_errors[0].get('message'),
                suggestions=static_errors[0].get('suggestions', [])
            )

        # Use ML model for deeper analysis if available
        if self.model is not None:
            ml_prediction = self._ml_predict(code, language)
            return ml_prediction
        else:
            # Fallback to heuristic-based prediction
            heuristic_prediction = self._heuristic_predict(code, language)
            return PredictionResult(
                file_path=str(file_path),
                language=language,
                error_type=heuristic_prediction['error_type'],
                confidence=heuristic_prediction['confidence'],
                error_message=heuristic_prediction.get('message'),
                suggestions=heuristic_prediction.get('suggestions', [])
            )

    def predict_batch(self, file_paths: List[str]) -> List[PredictionResult]:
        """
        Predict errors for multiple files

        Args:
            file_paths: List of file paths to analyze

        Returns:
            List of PredictionResult objects
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.predict_file(file_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error predicting {file_path}: {e}")
                # Add error result
                results.append(PredictionResult(
                    file_path=file_path,
                    language="unknown",
                    error_type=ErrorType.UNKNOWN,
                    confidence=0.0,
                    error_message=f"Prediction failed: {str(e)}"
                ))
        return results

    def _ml_predict(self, code: str, language: str) -> PredictionResult:
        """Use ML model to predict errors"""
        try:
            # Preprocess code for model input
            features = self.preprocessor.extract_features(code, language)

            if TORCH_AVAILABLE and hasattr(self.model, 'predict'):
                # PyTorch model
                with torch.no_grad():
                    if isinstance(features, dict):
                        # Convert features to tensor
                        input_tensor = torch.tensor([list(features.values())], dtype=torch.float32)
                    else:
                        input_tensor = torch.tensor(features, dtype=torch.float32)

                    predictions = self.model(input_tensor)
                    probabilities = torch.softmax(predictions, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()

            else:
                # Scikit-learn or other model
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba([features])[0]
                    if NUMPY_AVAILABLE:
                        predicted_class = np.argmax(probabilities)
                    else:
                        predicted_class = max(range(len(probabilities)), key=probabilities.__getitem__)
                    confidence = probabilities[predicted_class]
                else:
                    predicted_class = self.model.predict([features])[0]
                    confidence = 0.8  # Default confidence

            # Map prediction to error type
            error_types = list(ErrorType)
            if predicted_class < len(error_types):
                error_type = error_types[predicted_class]
            else:
                error_type = ErrorType.UNKNOWN

            return PredictionResult(
                file_path="",  # Will be filled by caller
                language=language,
                error_type=error_type,
                confidence=float(confidence)
            )

        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            return self._heuristic_predict(code, language)

    def _heuristic_predict(self, code: str, language: str) -> Dict:
        """
        Fallback heuristic-based error prediction when no ML model is available
        """
        predictions = {
            'error_type': ErrorType.NO_ERROR,
            'confidence': 0.6,
            'message': None,
            'suggestions': []
        }

        # Language-specific heuristics
        if language == 'python':
            predictions.update(self._python_heuristics(code))
        elif language in ['c', 'cpp']:
            predictions.update(self._c_cpp_heuristics(code))
        elif language == 'javascript':
            predictions.update(self._javascript_heuristics(code))

        return predictions

    def _python_heuristics(self, code: str) -> Dict:
        """Python-specific error detection heuristics"""
        issues = []

        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # Common Python issues
            if 'except:' in line_stripped and 'pass' in line_stripped:
                issues.append({
                    'type': ErrorType.LOGIC_ERROR,
                    'line': i,
                    'message': 'Bare except clause may hide errors',
                    'confidence': 0.7
                })

            if line_stripped.startswith('print') and '(' not in line:
                issues.append({
                    'type': ErrorType.SYNTAX_ERROR,
                    'line': i,
                    'message': 'Python 2 print syntax in Python 3',
                    'confidence': 0.8
                })

        if issues:
            return {
                'error_type': ErrorType(issues[0]['type']) if isinstance(issues[0]['type'], str) else issues[0]['type'],
                'confidence': issues[0]['confidence'],
                'message': issues[0]['message'],
                'suggestions': ['Review the flagged line for potential issues']
            }

        return {'error_type': ErrorType.NO_ERROR, 'confidence': 0.6}

    def _c_cpp_heuristics(self, code: str) -> Dict:
        """C/C++ specific error detection heuristics"""
        issues = []

        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # Memory management issues
            if 'malloc' in line and 'free' not in code:
                issues.append({
                    'type': ErrorType.MEMORY_ERROR,
                    'line': i,
                    'message': 'malloc without corresponding free',
                    'confidence': 0.7
                })

            # Buffer overflow potential
            if 'gets(' in line:
                issues.append({
                    'type': ErrorType.RUNTIME_ERROR,
                    'line': i,
                    'message': 'gets() is unsafe and deprecated',
                    'confidence': 0.9
                })

        if issues:
            return {
                'error_type': ErrorType(issues[0]['type']) if isinstance(issues[0]['type'], str) else issues[0]['type'],
                'confidence': issues[0]['confidence'],
                'message': issues[0]['message'],
                'suggestions': ['Consider using safer alternatives']
            }

        return {'error_type': ErrorType.NO_ERROR, 'confidence': 0.6}

    def _javascript_heuristics(self, code: str) -> Dict:
        """JavaScript specific error detection heuristics"""
        issues = []

        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # Common JS issues
            if '==' in line and '===' not in line:
                issues.append({
                    'type': ErrorType.LOGIC_ERROR,
                    'line': i,
                    'message': 'Use === instead of == for strict equality',
                    'confidence': 0.6
                })

            if 'var ' in line:
                issues.append({
                    'type': ErrorType.LOGIC_ERROR,
                    'line': i,
                    'message': 'Consider using let or const instead of var',
                    'confidence': 0.5
                })

        if issues:
            return {
                'error_type': ErrorType(issues[0]['type']) if isinstance(issues[0]['type'], str) else issues[0]['type'],
                'confidence': issues[0]['confidence'],
                'message': issues[0]['message'],
                'suggestions': ['Follow modern JavaScript best practices']
            }

        return {'error_type': ErrorType.NO_ERROR, 'confidence': 0.6}


def main():
    """Example usage of the ErrorDetectionModel"""
    predictor = ErrorDetectionModel()

    # Example: analyze a Python file
    if len(os.sys.argv) > 1:
        file_path = os.sys.argv[1]
        result = predictor.predict_file(file_path)

        print(f"File: {result.file_path}")
        print(f"Language: {result.language}")
        print(f"Predicted Error: {result.error_type.value}")
        print(f"Confidence: {result.confidence:.2f}")

        if result.error_message:
            print(f"Message: {result.error_message}")

        if result.suggestions:
            print("Suggestions:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")
    else:
        print("Usage: python error_predictor.py <file_path>")


if __name__ == "__main__":
    main()