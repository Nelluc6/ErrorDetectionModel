# Quick Reference Guide

## Docker Commands (Most Common)

### Setup
```bash
./docker-quickstart.sh              # Automated setup
# OR
docker-compose build                # Manual build
mkdir -p code_to_analyze reports    # Create directories
```

### Analyze Code
```bash
# Basic analysis
docker-compose run --rm error-detection analyze /code/*.py

# With HTML report
docker-compose run --rm error-detection \
  analyze /code/*.py --html /app/reports/report.html

# Try the example
docker-compose run --rm error-detection analyze /code/example.py

# Recursive (all files in directory)
docker-compose run --rm error-detection analyze -r /code/
```

### Advanced Options
```bash
# High confidence only
docker-compose run --rm error-detection \
  analyze /code/*.py --confidence 0.8

# Custom model
docker-compose run --rm error-detection \
  analyze -m /app/models/my_model.pkl /code/*.py

# Interactive shell
docker-compose run --rm error-detection bash
```

### View Reports
```bash
open reports/report.html            # macOS
xdg-open reports/report.html        # Linux
start reports/report.html           # Windows
```

## Local Commands (Without Docker)

### Analyze Code
```bash
python cli.py script.py                    # Single file
python cli.py *.py                         # Multiple files
python cli.py -r src/                      # Recursive
python cli.py --html report.html src/      # HTML report
python cli.py -m models/model.pkl *.py     # Custom model
```

## Directory Structure

```
code_to_analyze/    ← Put your code here
reports/            ← Reports saved here
models/             ← ML models here
```

## File Paths (Inside Docker Container)

- Your code: `/code/`
- Reports: `/app/reports/`
- Models: `/app/models/`

## Common Workflows

### Workflow 1: Analyze Your Project
```bash
# 1. Copy code
cp -r /path/to/project/*.py code_to_analyze/

# 2. Analyze
docker-compose run --rm error-detection analyze /code

# 3. Generate report
docker-compose run --rm error-detection \
  analyze /code --html /app/reports/report.html
```

### Workflow 2: Use Custom Model
```bash
# 1. Place model in models/
cp my_model.pkl models/

# 2. Analyze with model
docker-compose run --rm error-detection \
  analyze -m /app/models/my_model.pkl /code/*.py
```

### Workflow 3: Train a Model
```bash
# 1. Edit docker-compose.yml to mount CodeNet data
# 2. Run training
docker-compose --profile training run --rm training

# 3. Use the trained model
docker-compose run --rm error-detection \
  analyze -m /app/models/runtime_error_model.pkl /code
```

## Troubleshooting

### Container won't start
```bash
docker-compose logs error-detection
docker-compose build --no-cache
```

### Can't find files
```bash
# List files in container
docker-compose run --rm error-detection ls -la /code
```

### Permission errors
```bash
# Fix ownership (Linux/macOS)
sudo chown -R $(id -u):$(id -g) code_to_analyze/ reports/ models/
```

## Links

- **Full Docker Guide**: [DOCKER.md](DOCKER.md)
- **Main Documentation**: [README.md](README.md)
- **CodeNet Training**: [QUICKSTART_CODENET.md](QUICKSTART_CODENET.md)

## Need Help?

```bash
# Show CLI help
docker-compose run --rm error-detection help

# Interactive mode
docker-compose run --rm error-detection bash
```
