#!/bin/bash
# Quick Start Script for Error Detection Model (Docker)
# This script helps users get started quickly with the containerized version

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_info "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker from https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker is installed ($(docker --version))"
}

# Check if Docker Compose is installed
check_docker_compose() {
    print_info "Checking Docker Compose installation..."
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install it from https://docs.docker.com/compose/install/"
        exit 1
    fi
    print_success "Docker Compose is installed ($(docker-compose --version))"
}

# Build the Docker image
build_image() {
    print_info "Building Docker image... (this may take a few minutes)"
    if docker-compose build; then
        print_success "Docker image built successfully!"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Create necessary directories
setup_directories() {
    print_info "Setting up directories..."
    mkdir -p code_to_analyze reports models
    print_success "Directories created: code_to_analyze/, reports/, models/"
}

# Run example analysis
run_example() {
    print_info "Running example analysis..."
    if [ -f "code_to_analyze/example.py" ]; then
        print_info "Analyzing example.py..."
        docker-compose run --rm error-detection analyze /code/example.py
        print_success "Example analysis complete!"
    else
        print_warning "Example file not found. Skipping example analysis."
    fi
}

# Display usage instructions
show_usage() {
    cat << EOF

${GREEN}==================================================================
  Error Detection Model - Docker Quick Start Complete!
==================================================================${NC}

${BLUE}Next Steps:${NC}

1. ${YELLOW}Place your code files in the code_to_analyze/ directory:${NC}
   cp /path/to/your/code/*.py code_to_analyze/

2. ${YELLOW}Run analysis:${NC}
   docker-compose run --rm error-detection analyze /code

3. ${YELLOW}Generate HTML report:${NC}
   docker-compose run --rm error-detection analyze /code --html /app/reports/report.html
   open reports/report.html

4. ${YELLOW}Use a custom model:${NC}
   docker-compose run --rm error-detection analyze -m /app/models/your_model.pkl /code

5. ${YELLOW}Interactive shell:${NC}
   docker-compose run --rm error-detection bash

${BLUE}Common Commands:${NC}

  Analyze all Python files:
    docker-compose run --rm error-detection analyze /code/*.py

  Recursive analysis:
    docker-compose run --rm error-detection analyze -r /code/

  Filter by confidence:
    docker-compose run --rm error-detection analyze /code/*.py --confidence 0.8

  Generate JSON output:
    docker-compose run --rm error-detection analyze /code/*.py --json /app/reports/results.json

${BLUE}Documentation:${NC}
  - Full Docker guide: ${YELLOW}DOCKER.md${NC}
  - Main documentation: ${YELLOW}README.md${NC}
  - CodeNet training: ${YELLOW}QUICKSTART_CODENET.md${NC}

${GREEN}Happy coding!${NC}

EOF
}

# Main execution
main() {
    echo ""
    print_info "Error Detection Model - Docker Quick Start"
    echo ""

    # Check prerequisites
    check_docker
    check_docker_compose

    # Setup
    setup_directories
    build_image

    # Run example if requested
    if [ "$1" == "--run-example" ] || [ "$1" == "-e" ]; then
        run_example
    fi

    # Show usage
    show_usage
}

# Run main function
main "$@"
