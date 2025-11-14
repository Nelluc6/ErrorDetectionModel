# Docker Guide for Error Detection Model

This guide provides comprehensive instructions for using the Error Detection Model with Docker. The containerized version ensures a clean, isolated environment and prevents any accidental modifications to the codebase.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Building the Image](#building-the-image)
- [Usage Examples](#usage-examples)
- [Docker Compose](#docker-compose)
- [Volume Mounts](#volume-mounts)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Docker**: Version 20.10 or higher ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 1.29 or higher (usually included with Docker Desktop)

Verify installation:
```bash
docker --version
docker-compose --version
```

## Quick Start

Get up and running in 3 steps:

```bash
# 1. Clone the repository
git clone <repository-url>
cd ErrorDetectionModel

# 2. Build the Docker image
docker-compose build

# 3. Analyze your code
mkdir -p code_to_analyze reports
# Copy your code files to code_to_analyze/
docker-compose run --rm error-detection analyze /code
```

## Building the Image

### Using Docker Compose (Recommended)

```bash
docker-compose build
```

### Using Docker directly

```bash
docker build -t error-detection-model:latest .
```

The image is optimized for size and security:
- Based on Python 3.11 slim
- Runs as non-root user
- Multi-stage build for smaller size
- Includes all required dependencies

## Usage Examples

### 1. Analyze Code Files

#### Using Docker Compose

```bash
# Place your code in the code_to_analyze directory
mkdir -p code_to_analyze
cp /path/to/your/code/*.py code_to_analyze/

# Analyze all files
docker-compose run --rm error-detection analyze /code/*.py
```

#### Using Docker directly

```bash
docker run --rm \
  -v $(pwd)/code_to_analyze:/code:ro \
  error-detection-model:latest \
  analyze /code/*.py
```

### 2. Generate HTML Report

```bash
# Create reports directory
mkdir -p reports

# Generate HTML report
docker-compose run --rm error-detection \
  analyze /code/*.py --html /app/reports/error_report.html

# View the report in your browser
open reports/error_report.html
```

### 3. Analyze with Custom Options

```bash
# Filter by error type and confidence
docker-compose run --rm error-detection \
  analyze /code/*.py --errors-only --confidence 0.8

# Analyze recursively
docker-compose run --rm error-detection \
  analyze -r /code/

# Generate JSON output
docker-compose run --rm error-detection \
  analyze /code/*.py --json /app/reports/results.json
```

### 4. Use a Custom Trained Model

```bash
# Place your trained model in the models directory
cp /path/to/your/model.pkl models/

# Use the model for analysis
docker-compose run --rm error-detection \
  analyze -m /app/models/model.pkl /code/*.py
```

### 5. Interactive Shell Access

Sometimes you need to explore interactively:

```bash
# Start a bash shell in the container
docker-compose run --rm error-detection bash

# Inside the container:
python cli.py --help
python error_predictor.py /code/sample.py
exit
```

## Docker Compose

The `docker-compose.yml` file defines multiple service profiles:

### Main Service: `error-detection`

Default service for code analysis.

```bash
# Run analysis
docker-compose run --rm error-detection analyze /code/*.py

# View help
docker-compose run --rm error-detection help
```

### Training Service: `training`

For training models with CodeNet data (requires CodeNet dataset).

```bash
# First, update docker-compose.yml to mount your CodeNet path
# Uncomment and edit the volume line under the training service:
# - /path/to/Project_CodeNet:/data/Project_CodeNet:ro

# Start training with profile
docker-compose --profile training run --rm training

# Custom training
docker-compose --profile training run --rm training \
  python codenet_model_trainer.py \
  --output /app/models/my_model.pkl \
  --model-type gradient_boosting \
  --max-problems 200 \
  --languages Python "C++"
```

### GUI Service: `gui`

Launch the graphical interface (if available).

```bash
# Start the GUI service
docker-compose --profile gui up

# Access at http://localhost:8080
```

## Volume Mounts

The Docker setup uses read-only mounts to prevent accidental code modifications.

### Default Mounts

```yaml
volumes:
  - ./code_to_analyze:/code:ro          # Your code (read-only)
  - ./reports:/app/reports               # Output reports
  - ./models:/app/models                 # ML models
```

### Custom Mounts

Analyze code from any directory:

```bash
# Analyze code from a different location
docker run --rm \
  -v /path/to/your/project:/code:ro \
  -v $(pwd)/reports:/app/reports \
  error-detection-model:latest \
  analyze /code
```

### CodeNet Data Mount

For training, mount the CodeNet dataset:

```bash
docker run --rm \
  -v /path/to/Project_CodeNet:/data/Project_CodeNet:ro \
  -v $(pwd)/models:/app/models \
  error-detection-model:latest \
  train --max-problems 50
```

## Advanced Usage

### Environment Variables

Configure the container with environment variables:

```bash
# Set CodeNet path
docker-compose run --rm \
  -e CODENET_PATH=/data/Project_CodeNet \
  error-detection analyze /code

# Multiple environment variables
docker-compose run --rm \
  -e CODENET_PATH=/data/Project_CodeNet \
  -e PYTHONUNBUFFERED=1 \
  error-detection analyze /code
```

### Resource Limits

Control container resources:

```bash
# Limit CPU and memory
docker run --rm \
  --cpus="2" \
  --memory="4g" \
  -v $(pwd)/code_to_analyze:/code:ro \
  error-detection-model:latest \
  analyze /code
```

### Batch Processing

Process multiple projects:

```bash
#!/bin/bash
# analyze_projects.sh

PROJECTS=("project1" "project2" "project3")

for project in "${PROJECTS[@]}"; do
  echo "Analyzing $project..."
  docker run --rm \
    -v $(pwd)/$project:/code:ro \
    -v $(pwd)/reports:/app/reports \
    error-detection-model:latest \
    analyze /code --html /app/reports/${project}_report.html
done
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Code Error Detection

on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t error-detection .

      - name: Run error detection
        run: |
          docker run --rm \
            -v ${{ github.workspace }}:/code:ro \
            error-detection \
            analyze /code --json /tmp/results.json

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: error-detection-results
          path: /tmp/results.json
```

## Troubleshooting

### Permission Issues

If you encounter permission errors with volume mounts:

```bash
# Check file ownership
ls -la code_to_analyze/

# Fix permissions (on Linux)
sudo chown -R $(id -u):$(id -g) code_to_analyze/ reports/ models/
```

### Container Doesn't Start

```bash
# View container logs
docker-compose logs error-detection

# Check if image built successfully
docker images | grep error-detection

# Rebuild without cache
docker-compose build --no-cache
```

### Out of Memory

For large codebases or training:

```bash
# Increase Docker memory limit (Docker Desktop)
# Go to: Docker Desktop > Settings > Resources > Memory

# Or use command line limits
docker run --rm --memory="8g" \
  -v $(pwd)/code_to_analyze:/code:ro \
  error-detection-model:latest \
  analyze /code
```

### File Not Found Errors

Ensure paths are correct inside the container:

```bash
# List files in mounted volume
docker-compose run --rm error-detection ls -la /code

# Check mount point
docker-compose run --rm error-detection pwd
```

### Python Package Issues

If you need additional packages:

```bash
# Method 1: Extend the Dockerfile
# Create a custom Dockerfile:
FROM error-detection-model:latest
RUN pip install --no-cache-dir your-package

# Method 2: Install at runtime (not persistent)
docker-compose run --rm error-detection bash
pip install your-package
python cli.py ...
```

## Best Practices

1. **Always use read-only mounts** for source code:
   ```bash
   -v $(pwd)/code:/code:ro
   ```

2. **Keep models separate** from the container image:
   ```bash
   -v $(pwd)/models:/app/models
   ```

3. **Use docker-compose** for consistent environments:
   ```bash
   docker-compose run --rm error-detection analyze /code
   ```

4. **Clean up containers** with `--rm` flag:
   ```bash
   docker run --rm ...  # Container auto-deleted after exit
   ```

5. **Version your Docker images**:
   ```bash
   docker tag error-detection-model:latest error-detection-model:v1.0
   ```

6. **Scan for vulnerabilities**:
   ```bash
   docker scan error-detection-model:latest
   ```

## Docker Image Management

### Clean Up

```bash
# Remove unused containers
docker container prune

# Remove unused images
docker image prune

# Remove everything (careful!)
docker system prune -a
```

### Export/Import Images

```bash
# Save image to file
docker save error-detection-model:latest | gzip > error-detection.tar.gz

# Load image from file
docker load < error-detection.tar.gz
```

### Push to Registry

```bash
# Tag for registry
docker tag error-detection-model:latest username/error-detection-model:latest

# Push to Docker Hub
docker push username/error-detection-model:latest
```

## Getting Help

- View CLI help: `docker-compose run --rm error-detection help`
- Interactive mode: `docker-compose run --rm error-detection bash`
- Check logs: `docker-compose logs`

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Main README](README.md)
- [CodeNet Training Guide](QUICKSTART_CODENET.md)
