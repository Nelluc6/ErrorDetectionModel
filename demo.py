#!/usr/bin/env python3
"""
Demo Script for Error Detection Model

This script demonstrates the basic functionality of the error detection system
by analyzing a few sample code files with known issues.
"""

import tempfile
from pathlib import Path
from error_predictor import ErrorDetectionModel
from error_reporter import ErrorReporter


def create_demo_files():
    """Create sample files for demonstration"""
    temp_dir = Path(tempfile.mkdtemp(prefix="error_detection_demo_"))

    # Python file with issues
    python_code = '''
def problematic_function():
    try:
        result = eval("2 + 2")  # Dangerous eval usage
        print result  # Python 2 syntax error
    except:  # Bare except clause
        pass
    return result

# Missing return in function below
def incomplete_function():
    x = 10
    if x > 5:
        print("Greater than 5")
    # No return statement
'''

    # JavaScript file with issues
    js_code = '''
function checkValue(a, b) {
    var result;  // Should use let or const

    if (a == b) {  // Should use strict equality ===
        result = true;
    } else {
        result = false;
    }

    console.log("Debug:", result);  // Console statement in production
    return result;
}

// Undefined variable usage
function buggyFunction() {
    return someUndefinedVar + 10;
}
'''

    # C file with memory issues
    c_code = '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    char* buffer = malloc(100);  // Memory allocated
    char input[10];

    gets(input);  // Dangerous function - buffer overflow risk
    strcpy(buffer, input);  // Potential buffer overflow

    printf("Input: %s\\n", buffer);

    // Missing free(buffer) - memory leak
    return 0;
}
'''

    # Good Python file (no issues)
    good_python = '''
def safe_function():
    """A well-written function with proper error handling."""
    try:
        result = 2 + 2
        print(f"Result: {result}")
        return result
    except ValueError as e:
        print(f"Error occurred: {e}")
        return None

def calculate_sum(numbers):
    """Calculate sum of numbers with input validation."""
    if not isinstance(numbers, list):
        raise TypeError("Input must be a list")

    return sum(numbers)
'''

    # Create files
    files = [
        (temp_dir / "problematic.py", python_code),
        (temp_dir / "issues.js", js_code),
        (temp_dir / "unsafe.c", c_code),
        (temp_dir / "good_code.py", good_python)
    ]

    for file_path, content in files:
        file_path.write_text(content)

    return [str(f[0]) for f in files]


def run_demo():
    """Run the error detection demo"""
    print("🔍 Error Detection Model - Demo")
    print("=" * 50)

    # Create demo files
    print("Creating sample files with known issues...")
    demo_files = create_demo_files()

    print(f"Created {len(demo_files)} demo files:\n")
    for file_path in demo_files:
        print(f"  📄 {Path(file_path).name}")

    print("\n" + "=" * 50)

    # Initialize the error detection model
    print("Initializing Error Detection Model...")
    model = ErrorDetectionModel()  # No pre-trained model, uses heuristics
    reporter = ErrorReporter()

    # Analyze each file
    print("\nAnalyzing files for potential errors...")
    print("-" * 50)

    results = []

    for file_path in demo_files:
        file_name = Path(file_path).name
        print(f"\n📄 Analyzing {file_name}...")

        try:
            result = model.predict_file(file_path)
            results.append(result)

            # Display basic results
            error_status = "❌ ERRORS DETECTED" if result.error_type.value != "no_error" else "✅ NO ERRORS"
            print(f"   Status: {error_status}")
            print(f"   Language: {result.language}")
            print(f"   Error Type: {result.error_type.value}")
            print(f"   Confidence: {result.confidence:.2f}")

            if result.error_message:
                print(f"   Message: {result.error_message}")

            if result.suggestions:
                print(f"   Suggestions:")
                for suggestion in result.suggestions:
                    print(f"     • {suggestion}")

        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")

    # Generate summary report
    print("\n" + "=" * 50)
    print("📊 SUMMARY REPORT")
    print("=" * 50)

    reporter.print_summary_stats(results)

    # Generate detailed console report
    print("\n" + "=" * 50)
    print("📋 DETAILED REPORT")
    print("=" * 50)

    detailed_report = reporter.format_console_report(results)
    print(detailed_report)

    # Clean up demo files
    print("\n" + "=" * 50)
    print("🧹 Cleaning up demo files...")

    temp_dir = Path(demo_files[0]).parent
    for file_path in demo_files:
        Path(file_path).unlink()
    temp_dir.rmdir()

    print("Demo completed! ✨")
    print("\nTo analyze your own files, use:")
    print("  python cli.py your_file.py")
    print("  python cli.py -r your_project/")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()