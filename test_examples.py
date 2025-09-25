#!/usr/bin/env python3
"""
Test Examples and Validation

This module contains test code examples and validation functions
for the error detection system.
"""

import tempfile
from pathlib import Path
from typing import List, Dict

from error_predictor import ErrorDetectionModel, ErrorType
from error_reporter import ErrorReporter


# Test code samples with known issues
TEST_PYTHON_CODE = {
    "syntax_error": '''
def broken_function()  # Missing colon
    print("This will cause a syntax error")
    return True
''',

    "logic_error": '''
def risky_function():
    try:
        result = eval("2 + 2")  # Dangerous eval
        print result  # Python 2 syntax
    except:  # Bare except
        pass
    return result
''',

    "no_error": '''
def good_function():
    """A well-written function"""
    try:
        result = 2 + 2
        print(result)
        return result
    except ValueError as e:
        print(f"Error: {e}")
        return None
'''
}

TEST_JAVASCRIPT_CODE = {
    "logic_error": '''
function checkEquality(a, b) {
    var result;  // Should use let/const
    if (a == b) {  // Should use ===
        result = true;
    } else {
        result = false;
    }
    console.log("Result:", result);  // Should not be in production
    return result;
}
''',

    "no_error": '''
function checkEquality(a, b) {
    const result = a === b;
    return result;
}
'''
}

TEST_C_CODE = {
    "runtime_error": '''
#include <stdio.h>
#include <stdlib.h>

int main() {
    char* buffer = malloc(100);  // Missing free()
    char input[10];

    gets(input);  // Dangerous function
    strcpy(buffer, input);  // Potential buffer overflow

    printf(buffer);  // Format string vulnerability

    return 0;
}
''',

    "memory_error": '''
#include <stdio.h>
#include <stdlib.h>

int main() {
    int* ptr1 = malloc(sizeof(int) * 10);
    int* ptr2 = malloc(sizeof(int) * 20);

    // Use the memory
    ptr1[0] = 42;
    ptr2[0] = 84;

    free(ptr1);
    // Missing free(ptr2) - memory leak

    return 0;
}
''',

    "no_error": '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    const int size = 100;
    char* buffer = malloc(size);

    if (buffer == NULL) {
        return 1;
    }

    strncpy(buffer, "Hello, World!", size - 1);
    buffer[size - 1] = '\\0';

    printf("Message: %s\\n", buffer);

    free(buffer);
    return 0;
}
'''
}

TEST_CPP_CODE = {
    "memory_error": '''
#include <iostream>
#include <string>

int main() {
    int* numbers = new int[100];  // Missing delete[]
    std::string* name = new std::string("Test");  // Missing delete

    numbers[0] = 42;
    std::cout << *name << std::endl;

    // Memory leak - missing delete[] numbers and delete name
    return 0;
}
''',

    "no_error": '''
#include <iostream>
#include <vector>
#include <string>
#include <memory>

int main() {
    std::vector<int> numbers(100);  // Automatic memory management
    auto name = std::make_unique<std::string>("Test");  // Smart pointer

    numbers[0] = 42;
    std::cout << *name << std::endl;

    return 0;  // Automatic cleanup
}
'''
}


def create_test_files() -> List[str]:
    """
    Create temporary test files with known issues

    Returns:
        List of temporary file paths
    """
    test_files = []
    temp_dir = Path(tempfile.mkdtemp(prefix="error_detection_test_"))

    # Python test files
    for name, code in TEST_PYTHON_CODE.items():
        file_path = temp_dir / f"test_python_{name}.py"
        file_path.write_text(code)
        test_files.append(str(file_path))

    # JavaScript test files
    for name, code in TEST_JAVASCRIPT_CODE.items():
        file_path = temp_dir / f"test_js_{name}.js"
        file_path.write_text(code)
        test_files.append(str(file_path))

    # C test files
    for name, code in TEST_C_CODE.items():
        file_path = temp_dir / f"test_c_{name}.c"
        file_path.write_text(code)
        test_files.append(str(file_path))

    # C++ test files
    for name, code in TEST_CPP_CODE.items():
        file_path = temp_dir / f"test_cpp_{name}.cpp"
        file_path.write_text(code)
        test_files.append(str(file_path))

    return test_files


def run_validation_tests() -> Dict[str, bool]:
    """
    Run validation tests on the error detection system

    Returns:
        Dictionary of test results
    """
    results = {}

    # Initialize the error detection model (without pre-trained model)
    model = ErrorDetectionModel()
    reporter = ErrorReporter()

    # Create test files
    test_files = create_test_files()

    print("Running validation tests...")
    print("=" * 50)

    # Test each file
    predictions = []
    for file_path in test_files:
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name

        try:
            result = model.predict_file(file_path)
            predictions.append(result)

            # Determine if prediction is reasonable
            expected_error = not file_name.endswith("_no_error.py") and \
                           not file_name.endswith("_no_error.js") and \
                           not file_name.endswith("_no_error.c") and \
                           not file_name.endswith("_no_error.cpp")

            actual_error = result.error_type != ErrorType.NO_ERROR

            test_passed = (expected_error == actual_error) or result.confidence > 0.5

            results[file_name] = test_passed

            status = "✓ PASS" if test_passed else "✗ FAIL"
            print(f"{status} {file_name}")
            print(f"      Language: {result.language}")
            print(f"      Predicted: {result.error_type.value}")
            print(f"      Confidence: {result.confidence:.2f}")

            if result.error_message:
                print(f"      Message: {result.error_message}")

            print()

        except Exception as e:
            results[file_name] = False
            print(f"✗ FAIL {file_name} - Exception: {e}")
            print()

    # Generate test report
    print("Test Summary")
    print("=" * 50)

    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests

    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

    # Generate full report
    print("\nFull Report:")
    print("=" * 50)
    report = reporter.format_console_report(predictions)
    print(report)

    # Clean up test files
    temp_dir = Path(test_files[0]).parent
    for file_path in test_files:
        Path(file_path).unlink()
    temp_dir.rmdir()

    return results


def test_individual_components():
    """Test individual components of the system"""
    from code_preprocessor import CodePreprocessor

    print("Testing Code Preprocessor...")
    print("-" * 30)

    preprocessor = CodePreprocessor()

    # Test language detection
    test_cases = [
        ("script.py", "python"),
        ("app.js", "javascript"),
        ("main.c", "c"),
        ("program.cpp", "cpp"),
        ("unknown.txt", "unknown")
    ]

    for filename, expected in test_cases:
        detected = preprocessor.detect_language(filename)
        status = "✓" if detected == expected else "✗"
        print(f"{status} {filename} -> {detected} (expected: {expected})")

    print("\nTesting Static Analysis...")
    print("-" * 30)

    # Test Python static analysis
    python_code = 'print "Hello"  # Python 2 syntax\nexcept:\n    pass'
    issues = preprocessor.static_analysis(python_code, 'python')
    print(f"Python issues found: {len(issues)}")

    # Test feature extraction
    print("\nTesting Feature Extraction...")
    print("-" * 30)

    features = preprocessor.extract_features(python_code, 'python')
    print(f"Extracted {len(features)} features")
    print(f"Sample features: {list(features.keys())[:5]}")


def main():
    """Main testing function"""
    print("Error Detection Model - Validation Tests")
    print("=" * 60)

    # Test individual components
    test_individual_components()

    print("\n" + "=" * 60)

    # Run validation tests
    results = run_validation_tests()

    # Return exit code based on results
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())