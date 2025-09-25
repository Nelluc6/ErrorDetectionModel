#!/usr/bin/env python3
"""
CLI Tool for Error Detection Model

Command-line interface for predicting errors in code files using static analysis
and machine learning models.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import logging

# Add the current directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from error_predictor import ErrorDetectionModel, ErrorType
from error_reporter import ErrorReporter


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )


def find_code_files(paths: List[str], recursive: bool = False) -> List[str]:
    """
    Find all code files in the given paths

    Args:
        paths: List of file/directory paths
        recursive: Whether to search directories recursively

    Returns:
        List of code file paths
    """
    code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.c', '.cpp', '.cc',
                      '.cxx', '.c++', '.h', '.hpp', '.java', '.go', '.rs',
                      '.php', '.rb', '.swift', '.kt'}

    code_files = []

    for path_str in paths:
        path = Path(path_str)

        if path.is_file():
            if path.suffix.lower() in code_extensions:
                code_files.append(str(path))
            else:
                print(f"Warning: {path} is not a recognized code file type", file=sys.stderr)

        elif path.is_dir():
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"

            for file_path in path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in code_extensions:
                    code_files.append(str(file_path))

        else:
            print(f"Warning: {path} does not exist", file=sys.stderr)

    return code_files


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Predict potential errors in code files using static analysis and ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s script.py                    # Analyze single file
  %(prog)s *.py                         # Analyze multiple files
  %(prog)s src/                         # Analyze directory (non-recursive)
  %(prog)s -r project/                  # Analyze directory recursively
  %(prog)s -m model.pkl script.py       # Use specific ML model
  %(prog)s --json report.json *.py     # Save results as JSON
  %(prog)s --html report.html src/     # Generate HTML report
        """
    )

    # Input arguments
    parser.add_argument(
        'paths',
        nargs='+',
        help='Code files or directories to analyze'
    )

    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Search directories recursively'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        help='Path to trained ML model file (.pkl, .pt, .pth, or model directory)'
    )

    # Output options
    parser.add_argument(
        '--json',
        type=str,
        help='Save results to JSON file'
    )

    parser.add_argument(
        '--csv',
        type=str,
        help='Save results to CSV file'
    )

    parser.add_argument(
        '--html',
        type=str,
        help='Generate HTML report'
    )

    parser.add_argument(
        '--viz',
        type=str,
        help='Directory to save visualization charts'
    )

    # Filtering options
    parser.add_argument(
        '--lang',
        type=str,
        choices=['python', 'javascript', 'c', 'cpp', 'java', 'go', 'rust'],
        help='Filter by programming language'
    )

    parser.add_argument(
        '--errors-only',
        action='store_true',
        help='Show only files with potential errors'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.0,
        help='Minimum confidence threshold (0.0-1.0)'
    )

    # Display options
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress console output (except errors)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )

    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary statistics'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Find code files
        code_files = find_code_files(args.paths, args.recursive)

        if not code_files:
            print("Error: No code files found to analyze", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print(f"Found {len(code_files)} code files to analyze")

        # Initialize the error detection model
        model = ErrorDetectionModel(args.model)
        reporter = ErrorReporter()

        # Analyze files
        if not args.quiet:
            print("Analyzing files...")

        results = []
        for i, file_path in enumerate(code_files, 1):
            if not args.quiet:
                print(f"  [{i}/{len(code_files)}] {file_path}", end="", flush=True)

            try:
                result = model.predict_file(file_path)
                results.append(result)

                if not args.quiet:
                    status = "✓" if result.error_type == ErrorType.NO_ERROR else "⚠"
                    print(f" {status}")

            except Exception as e:
                if not args.quiet:
                    print(f" ✗ (Error: {e})")
                logging.error(f"Failed to analyze {file_path}: {e}")

        # Filter results
        filtered_results = results

        if args.lang:
            filtered_results = [r for r in filtered_results if r.language == args.lang]

        if args.errors_only:
            filtered_results = [r for r in filtered_results if r.error_type.value != 'no_error']

        if args.confidence > 0:
            filtered_results = [r for r in filtered_results if r.confidence >= args.confidence]

        # Generate reports
        if not args.quiet and not args.summary_only:
            print("\n" + reporter.format_console_report(filtered_results))

        if args.summary_only and not args.quiet:
            reporter.print_summary_stats(filtered_results)

        # Save to files
        if args.json:
            reporter.save_json_report(filtered_results, args.json)
            if not args.quiet:
                print(f"JSON report saved to {args.json}")

        if args.csv:
            reporter.save_csv_report(filtered_results, args.csv)
            if not args.quiet:
                print(f"CSV report saved to {args.csv}")

        if args.html:
            reporter.generate_html_report(filtered_results, args.html)
            if not args.quiet:
                print(f"HTML report saved to {args.html}")

        if args.viz:
            reporter.create_visualization(filtered_results, args.viz)
            if not args.quiet:
                print(f"Visualizations saved to {args.viz}")

        # Exit with appropriate code
        error_count = sum(1 for r in filtered_results if r.error_type.value != 'no_error')
        sys.exit(1 if error_count > 0 else 0)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()