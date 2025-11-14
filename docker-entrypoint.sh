#!/bin/bash
# Error Detection Model - Docker Entrypoint Script
# This script provides convenient commands for common tasks

set -e

# Function to display usage
show_usage() {
    cat << EOF
Error Detection Model - Docker Container

Usage:
  analyze <file_or_dir>     Analyze code files or directories
  train                      Train a model using CodeNet data
  help                       Show CLI help
  bash                       Start interactive bash shell

Examples:
  docker run --rm -v \$(pwd)/mycode:/code error-detection-model analyze /code
  docker run --rm -v \$(pwd)/models:/app/models error-detection-model train
  docker-compose run error-detection analyze /code/*.py --html /app/reports/report.html

EOF
}

# Main command handling
case "$1" in
    analyze)
        shift
        echo "Analyzing code: $@"
        exec python cli.py "$@"
        ;;

    train)
        shift
        echo "Training model..."
        if [ -z "$1" ]; then
            # Default training configuration
            exec python codenet_model_trainer.py \
                --output /app/models/runtime_error_model.pkl \
                --model-type random_forest \
                --max-problems 50 \
                --languages Python
        else
            exec python codenet_model_trainer.py "$@"
        fi
        ;;

    help)
        exec python cli.py --help
        ;;

    bash)
        exec /bin/bash
        ;;

    --help|-h)
        show_usage
        ;;

    *)
        # If command starts with python, execute it directly
        if [[ "$1" == "python"* ]] || [[ "$1" == "cli.py"* ]]; then
            exec "$@"
        else
            # Otherwise, pass to CLI
            exec python cli.py "$@"
        fi
        ;;
esac
