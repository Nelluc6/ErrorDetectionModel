# Quick Start Guide: Training Runtime Error Models with CodeNet

This guide will walk you through training a runtime error prediction model using IBM's Project CodeNet dataset.

## Prerequisites

1. **Python 3.7+** installed
2. **Project CodeNet dataset** downloaded from https://github.com/IBM/Project_CodeNet

## Step 1: Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Core numerical operations
- `scikit-learn` - Machine learning models
- `pandas` - Data manipulation

### Set CodeNet Path

```bash
export CODENET_PATH=/path/to/Project_CodeNet
```

Or on Windows:
```cmd
set CODENET_PATH=C:\path\to\Project_CodeNet
```

### Verify Setup

```python
python -c "from codenet_data_loader import CodeNetDataLoader; loader = CodeNetDataLoader(); print('Setup OK!' if loader.validate_dataset() else 'Check CODENET_PATH')"
```

## Step 2: Train Your First Model

### Quick Training (Small Dataset)

Train on a small subset to test the pipeline:

```bash
python codenet_model_trainer.py \
    --output models/test_model.pkl \
    --model-type random_forest \
    --max-problems 10 \
    --limit-per-problem 20 \
    --languages Python
```

This will:
- Load 10 problems
- Take up to 20 submissions per problem
- Focus on Python code only
- Save model to `models/test_model.pkl`
- Take ~1-2 minutes

### Production Training (Full Dataset)

For a production-ready model:

```bash
python codenet_model_trainer.py \
    --output models/runtime_error_model.pkl \
    --model-type random_forest \
    --max-problems 500 \
    --limit-per-problem 100 \
    --languages Python "C++" Java
```

This will:
- Train on 500 problems (substantial dataset)
- Use multiple languages
- Take 30-60 minutes depending on hardware
- Produce a robust model

## Step 3: Use Your Model

### Command Line

Analyze a single file:
```bash
python cli.py -m models/runtime_error_model.pkl my_code.py
```

Analyze multiple files:
```bash
python cli.py -m models/runtime_error_model.pkl src/*.py
```

Generate HTML report:
```bash
python cli.py -m models/runtime_error_model.pkl --html report.html src/
```

### Python API

```python
from error_predictor import ErrorDetectionModel

# Load your trained model
model = ErrorDetectionModel(model_path="models/runtime_error_model.pkl")

# Predict on a file
result = model.predict_file("my_script.py")

print(f"Error Type: {result.error_type.value}")
print(f"Confidence: {result.confidence:.2%}")
if result.error_message:
    print(f"Message: {result.error_message}")
```

## Step 4: Evaluate Model Performance

After training, check the metrics file:

```bash
cat models/training_metrics.json
```

This shows:
- Training accuracy
- Test accuracy
- Precision, recall, F1 scores
- Dataset statistics

## Model Types Comparison

| Model Type | Speed | Accuracy | Memory | Best For |
|------------|-------|----------|---------|----------|
| `random_forest` | Fast | High | Medium | General use (default) |
| `gradient_boosting` | Medium | Highest | Medium | Maximum accuracy |
| `logistic_regression` | Very Fast | Medium | Low | Quick iterations |
| `svm` | Slow | High | High | Small datasets |

## Advanced Usage

### Custom Data Loading

```python
from codenet_data_loader import CodeNetDataLoader
from codenet_model_trainer import RuntimeErrorPredictor

# Load specific data
loader = CodeNetDataLoader()
samples = loader.load_dataset(
    languages=["Python", "C++"],
    statuses=["Runtime Error", "Accepted", "Time Limit Exceeded"],
    max_problems=100,
    limit_per_problem=50
)

print(f"Loaded {len(samples)} samples")

# Check dataset balance
stats = loader.get_dataset_statistics(samples)
print(f"Label distribution: {stats['labels']}")

# Create splits
train, val, test = loader.create_training_split(samples, train_ratio=0.7)

# Train model
predictor = RuntimeErrorPredictor(model_type="random_forest")
metrics = predictor.train(train, val)

# Evaluate
test_metrics = predictor.evaluate(test)
print(f"Test accuracy: {test_metrics['accuracy']:.2%}")

# Save
predictor.save_model("models/custom_model.pkl")
```

### Using Pre-extracted JSONL Files

If you have pre-extracted data from `extract_codenet_split.py`:

```python
from codenet_data_loader import CodeNetDataLoader
from codenet_model_trainer import RuntimeErrorPredictor

loader = CodeNetDataLoader()

# Load from JSONL
train_samples = loader.load_from_jsonl("data/codenet_extract/train.jsonl")
val_samples = loader.load_from_jsonl("data/codenet_extract/valid.jsonl")
test_samples = loader.load_from_jsonl("data/codenet_extract/test.jsonl")

# Train
predictor = RuntimeErrorPredictor(model_type="gradient_boosting")
predictor.train(train_samples, val_samples)
predictor.save_model("models/my_model.pkl")
```

## Troubleshooting

### "CODENET_PATH not set"
```bash
export CODENET_PATH=/path/to/Project_CodeNet
```

### "No samples loaded"
- Check that CODENET_PATH points to the root directory
- Verify the directory contains `data/` and `metadata/` subdirectories
- Try loading a specific problem to debug:
```python
from codenet_data_loader import CodeNetDataLoader
loader = CodeNetDataLoader()
samples = loader.load_problem_samples("p00000")
print(f"Loaded {len(samples)} samples from p00000")
```

### "Import Error: sklearn"
```bash
pip install scikit-learn pandas
```

### Low Accuracy
- Increase `--max-problems` to use more training data
- Try `--model-type gradient_boosting` for better accuracy
- Ensure dataset is balanced across error types
- Filter to specific languages with `--languages`

## Next Steps

1. **Experiment with hyperparameters**: Edit `codenet_model_trainer.py` to tune model parameters
2. **Train language-specific models**: Use `--languages Python` for better Python-specific accuracy
3. **Integrate with CI/CD**: Add error detection to your build pipeline
4. **Fine-tune features**: Modify `code_preprocessor.py` to add domain-specific features

## Resources

- [Project CodeNet GitHub](https://github.com/IBM/Project_CodeNet)
- [Project CodeNet Paper](https://arxiv.org/abs/2105.12655)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## Getting Help

If you encounter issues:
1. Check the main README.md
2. Verify your CODENET_PATH is set correctly
3. Try the validation script: `python codenet_data_loader.py`
4. Review training logs for specific error messages
