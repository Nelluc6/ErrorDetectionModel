"""
CodeNet Data Loader Module

This module handles loading and preprocessing data from IBM's Project CodeNet dataset
for training runtime error prediction models.
"""

import os
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
import logging


@dataclass
class CodeNetSample:
    """Represents a single code sample from Project CodeNet"""
    problem_id: str
    submission_id: str
    language: str
    status: str  # e.g., "Accepted", "Runtime Error", "Time Limit Exceeded"
    code: str
    cpu_time: Optional[float] = None
    memory: Optional[int] = None
    code_size: Optional[int] = None
    accuracy: Optional[float] = None
    label: Optional[str] = None  # Mapped error category


class CodeNetDataLoader:
    """
    Loads and processes data from IBM's Project CodeNet dataset
    for runtime error prediction
    """

    def __init__(self, codenet_path: Optional[str] = None):
        """
        Initialize the CodeNet data loader

        Args:
            codenet_path: Path to Project_CodeNet directory (defaults to env variable)
        """
        self.logger = logging.getLogger(__name__)

        # Set CodeNet path from parameter or environment variable
        if codenet_path:
            self.codenet_path = Path(codenet_path)
        elif "CODENET_PATH" in os.environ:
            self.codenet_path = Path(os.environ["CODENET_PATH"])
        else:
            # Try local data directory
            self.codenet_path = Path(__file__).parent / "data" / "Project_CodeNet"

        self.metadata_dir = self.codenet_path / "metadata"
        self.data_dir = self.codenet_path / "data"

        # Status to label mapping
        self.status_mapping = {
            "Accepted": "no_error",
            "Runtime Error": "runtime_error",
            "Time Limit Exceeded": "timeout",
            "Memory Limit Exceeded": "memory_error",
            "Compile Error": "compile_error",
            "Wrong Answer": "logic_error",
            "Output Limit Exceeded": "runtime_error",
            "Judge Not Available": "unknown",
            "Internal Error": "unknown",
        }

    def validate_dataset(self) -> bool:
        """
        Validate that the CodeNet dataset is accessible

        Returns:
            True if dataset is valid, False otherwise
        """
        if not self.codenet_path.exists():
            self.logger.error(f"CodeNet path does not exist: {self.codenet_path}")
            return False

        if not self.metadata_dir.exists():
            self.logger.error(f"Metadata directory not found: {self.metadata_dir}")
            return False

        if not self.data_dir.exists():
            self.logger.error(f"Data directory not found: {self.data_dir}")
            return False

        return True

    def map_status_to_label(self, status: str) -> str:
        """
        Map CodeNet status to error category label

        Args:
            status: Status string from CodeNet metadata

        Returns:
            Mapped label string
        """
        return self.status_mapping.get(status, "unknown")

    def load_problem_metadata(self, problem_id: str) -> pd.DataFrame:
        """
        Load metadata for a specific problem

        Args:
            problem_id: Problem ID (e.g., "p00000")

        Returns:
            DataFrame with submission metadata
        """
        metadata_file = self.metadata_dir / f"{problem_id}.csv"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        df = pd.read_csv(metadata_file)

        # Add label column
        if 'status' in df.columns:
            df['label'] = df['status'].apply(self.map_status_to_label)

        return df

    def load_submission_code(self, problem_id: str, submission_id: str,
                            language: str, filename: str) -> Optional[str]:
        """
        Load source code for a specific submission

        Args:
            problem_id: Problem ID
            submission_id: Submission ID
            language: Programming language
            filename: Filename with extension

        Returns:
            Source code as string, or None if not found
        """
        code_path = self.data_dir / problem_id / language / filename

        if not code_path.exists():
            self.logger.warning(f"Code file not found: {code_path}")
            return None

        try:
            # Try UTF-8 first
            return code_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                return code_path.read_text(encoding='latin-1', errors='ignore')
            except Exception as e:
                self.logger.error(f"Failed to read {code_path}: {e}")
                return None

    def load_problem_samples(self, problem_id: str,
                            languages: Optional[List[str]] = None,
                            statuses: Optional[List[str]] = None,
                            limit: Optional[int] = None) -> List[CodeNetSample]:
        """
        Load all samples for a specific problem

        Args:
            problem_id: Problem ID to load
            languages: Filter by languages (e.g., ["Python", "C++"])
            statuses: Filter by statuses (e.g., ["Accepted", "Runtime Error"])
            limit: Maximum number of samples to load

        Returns:
            List of CodeNetSample objects
        """
        samples = []

        try:
            metadata = self.load_problem_metadata(problem_id)

            # Apply filters
            if languages:
                metadata = metadata[metadata['language'].isin(languages)]

            if statuses:
                metadata = metadata[metadata['status'].isin(statuses)]

            # Apply limit
            if limit:
                metadata = metadata.head(limit)

            for _, row in metadata.iterrows():
                submission_id = row['submission_id']
                language = row['language']
                status = row['status']
                filename_ext = row['filename_ext']

                filename = f"{submission_id}.{filename_ext}"
                code = self.load_submission_code(problem_id, submission_id, language, filename)

                if code is not None:
                    sample = CodeNetSample(
                        problem_id=problem_id,
                        submission_id=submission_id,
                        language=language,
                        status=status,
                        code=code,
                        cpu_time=row.get('cpu_time'),
                        memory=row.get('memory'),
                        code_size=row.get('code_size'),
                        accuracy=row.get('accuracy'),
                        label=self.map_status_to_label(status)
                    )
                    samples.append(sample)

        except Exception as e:
            self.logger.error(f"Error loading samples for {problem_id}: {e}")

        return samples

    def load_dataset(self, problem_ids: Optional[List[str]] = None,
                    languages: Optional[List[str]] = None,
                    statuses: Optional[List[str]] = None,
                    limit_per_problem: Optional[int] = None,
                    max_problems: Optional[int] = None) -> List[CodeNetSample]:
        """
        Load dataset from multiple problems

        Args:
            problem_ids: List of problem IDs to load (loads all if None)
            languages: Filter by languages
            statuses: Filter by statuses
            limit_per_problem: Max samples per problem
            max_problems: Max number of problems to process

        Returns:
            List of CodeNetSample objects
        """
        all_samples = []

        # Get list of problems
        if problem_ids is None:
            # Get all problem metadata files
            if not self.metadata_dir.exists():
                self.logger.error(f"Metadata directory not found: {self.metadata_dir}")
                return []

            problem_files = sorted(self.metadata_dir.glob("p*.csv"))
            problem_ids = [f.stem for f in problem_files]

            if max_problems:
                problem_ids = problem_ids[:max_problems]

        self.logger.info(f"Loading data from {len(problem_ids)} problems...")

        for i, problem_id in enumerate(problem_ids, 1):
            if i % 10 == 0:
                self.logger.info(f"Processed {i}/{len(problem_ids)} problems, {len(all_samples)} samples loaded")

            samples = self.load_problem_samples(
                problem_id,
                languages=languages,
                statuses=statuses,
                limit=limit_per_problem
            )
            all_samples.extend(samples)

        self.logger.info(f"Total samples loaded: {len(all_samples)}")
        return all_samples

    def load_from_jsonl(self, jsonl_path: str) -> List[CodeNetSample]:
        """
        Load samples from a JSONL file (e.g., from extract_codenet_split.py)

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            List of CodeNetSample objects
        """
        samples = []
        jsonl_path = Path(jsonl_path)

        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        with jsonl_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    sample = CodeNetSample(
                        problem_id=data['problem_id'],
                        submission_id=data['submission_id'],
                        language=data['language'],
                        status=data['status'],
                        code=data['code'],
                        label=data.get('label', self.map_status_to_label(data['status']))
                    )
                    samples.append(sample)
                except Exception as e:
                    self.logger.warning(f"Failed to parse line: {e}")

        self.logger.info(f"Loaded {len(samples)} samples from {jsonl_path}")
        return samples

    def get_dataset_statistics(self, samples: List[CodeNetSample]) -> Dict:
        """
        Get statistics about a dataset

        Args:
            samples: List of CodeNetSample objects

        Returns:
            Dictionary with dataset statistics
        """
        if not samples:
            return {}

        stats = {
            'total_samples': len(samples),
            'languages': {},
            'statuses': {},
            'labels': {},
            'avg_code_size': 0,
        }

        total_size = 0

        for sample in samples:
            # Count languages
            stats['languages'][sample.language] = stats['languages'].get(sample.language, 0) + 1

            # Count statuses
            stats['statuses'][sample.status] = stats['statuses'].get(sample.status, 0) + 1

            # Count labels
            if sample.label:
                stats['labels'][sample.label] = stats['labels'].get(sample.label, 0) + 1

            # Code size
            total_size += len(sample.code)

        stats['avg_code_size'] = total_size / len(samples) if samples else 0

        return stats

    def create_training_split(self, samples: List[CodeNetSample],
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             random_seed: int = 42) -> Tuple[List, List, List]:
        """
        Split samples into train/validation/test sets

        Args:
            samples: List of samples to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        import random
        random.seed(random_seed)

        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)

        # Calculate split points
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_samples = shuffled[:train_end]
        val_samples = shuffled[train_end:val_end]
        test_samples = shuffled[val_end:]

        self.logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

        return train_samples, val_samples, test_samples


def main():
    """Example usage of CodeNet data loader"""
    logging.basicConfig(level=logging.INFO)

    loader = CodeNetDataLoader()

    # Validate dataset
    if not loader.validate_dataset():
        print(f"Please set CODENET_PATH environment variable or place data in {loader.codenet_path}")
        return

    print("Loading sample data...")

    # Load a small sample focusing on runtime errors
    samples = loader.load_dataset(
        languages=["Python", "C++"],
        statuses=["Runtime Error", "Accepted"],
        limit_per_problem=10,
        max_problems=5
    )

    # Show statistics
    stats = loader.get_dataset_statistics(samples)
    print(f"\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Languages: {stats['languages']}")
    print(f"Statuses: {stats['statuses']}")
    print(f"Labels: {stats['labels']}")
    print(f"Avg code size: {stats['avg_code_size']:.1f} characters")

    # Show example
    if samples:
        print(f"\nExample sample:")
        sample = samples[0]
        print(f"Problem: {sample.problem_id}")
        print(f"Language: {sample.language}")
        print(f"Status: {sample.status}")
        print(f"Label: {sample.label}")
        print(f"Code preview: {sample.code[:200]}...")


if __name__ == "__main__":
    main()
