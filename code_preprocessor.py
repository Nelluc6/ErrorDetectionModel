"""
Code Preprocessor Module

This module handles preprocessing of code files for error detection,
including language detection, static analysis, and feature extraction.
"""

import ast
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

try:
    import pylint.lint
    import pylint.reporters.text
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

try:
    import esprima
    ESPRIMA_AVAILABLE = True
except ImportError:
    ESPRIMA_AVAILABLE = False


class CodePreprocessor:
    """
    Handles code preprocessing tasks including language detection,
    static analysis, and feature extraction for ML models.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Language file extensions mapping
        self.language_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c++': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }

        # Common error patterns by language
        self.error_patterns = {
            'python': [
                (r'^\s*print\s+[^(]', 'Python 2 print syntax'),
                (r'except:\s*$', 'Bare except clause'),
                (r'eval\s*\(', 'Unsafe eval usage'),
                (r'exec\s*\(', 'Unsafe exec usage'),
                (r'import\s+\*', 'Wildcard import'),
            ],
            'javascript': [
                (r'==(?!=)', 'Use === instead of =='),
                (r'!=(?!=)', 'Use !== instead of !='),
                (r'\bvar\b', 'Consider using let or const'),
                (r'eval\s*\(', 'Unsafe eval usage'),
            ],
            'c': [
                (r'gets\s*\(', 'Unsafe gets function'),
                (r'strcpy\s*\(', 'Consider using strncpy'),
                (r'sprintf\s*\(', 'Consider using snprintf'),
                (r'malloc.*(?!.*free)', 'Potential memory leak'),
            ],
            'cpp': [
                (r'gets\s*\(', 'Unsafe gets function'),
                (r'strcpy\s*\(', 'Consider using safe alternatives'),
                (r'new\s+.*(?!.*delete)', 'Potential memory leak'),
                (r'#include\s*<.*\.h>', 'Use C++ headers instead of C'),
            ]
        }

    def detect_language(self, file_path: Union[str, Path]) -> str:
        """
        Detect programming language from file extension

        Args:
            file_path: Path to the code file

        Returns:
            Detected language as string
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        return self.language_extensions.get(extension, 'unknown')

    def static_analysis(self, code: str, language: str) -> List[Dict[str, Any]]:
        """
        Perform static analysis on code to detect obvious errors

        Args:
            code: Source code as string
            language: Programming language

        Returns:
            List of detected issues
        """
        issues = []

        if language == 'python':
            issues.extend(self._python_static_analysis(code))
        elif language == 'javascript':
            issues.extend(self._javascript_static_analysis(code))
        elif language in ['c', 'cpp']:
            issues.extend(self._c_cpp_static_analysis(code))

        # Common pattern-based analysis for all languages
        issues.extend(self._pattern_based_analysis(code, language))

        return issues

    def _python_static_analysis(self, code: str) -> List[Dict[str, Any]]:
        """Python-specific static analysis using AST"""
        issues = []

        try:
            # Parse the AST
            tree = ast.parse(code)

            # Check for common issues
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append({
                        'type': 'logic_error',
                        'line': node.lineno,
                        'message': 'Bare except clause catches all exceptions',
                        'suggestions': ['Specify specific exception types to catch']
                    })

                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == 'eval':
                        issues.append({
                            'type': 'runtime_error',
                            'line': node.lineno,
                            'message': 'Use of eval() is dangerous',
                            'suggestions': ['Consider safer alternatives to eval()']
                        })

                elif isinstance(node, ast.ImportFrom) and node.module is None:
                    issues.append({
                        'type': 'logic_error',
                        'line': node.lineno,
                        'message': 'Relative import without package',
                        'suggestions': ['Use absolute imports or ensure proper package structure']
                    })

        except SyntaxError as e:
            issues.append({
                'type': 'syntax_error',
                'line': e.lineno or 1,
                'message': f'Python syntax error: {e.msg}',
                'suggestions': ['Fix the syntax error before proceeding']
            })

        # Use pylint if available
        if PYLINT_AVAILABLE:
            issues.extend(self._run_pylint(code))

        return issues

    def _javascript_static_analysis(self, code: str) -> List[Dict[str, Any]]:
        """JavaScript-specific static analysis"""
        issues = []

        try:
            if ESPRIMA_AVAILABLE:
                # Use esprima for JavaScript parsing
                try:
                    esprima.parse(code)
                except esprima.Error as e:
                    issues.append({
                        'type': 'syntax_error',
                        'line': getattr(e, 'lineNumber', 1),
                        'message': f'JavaScript syntax error: {e.description}',
                        'suggestions': ['Fix the syntax error']
                    })

            # Basic regex-based checks
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()

                # Check for console.log in production
                if 'console.log' in line_stripped:
                    issues.append({
                        'type': 'logic_error',
                        'line': i,
                        'message': 'console.log statement found',
                        'suggestions': ['Remove console.log before production deployment']
                    })

                # Check for var usage
                if re.search(r'\bvar\s+\w+', line_stripped):
                    issues.append({
                        'type': 'logic_error',
                        'line': i,
                        'message': 'Use let or const instead of var',
                        'suggestions': ['Replace var with let or const for block scoping']
                    })

        except Exception as e:
            self.logger.error(f"JavaScript static analysis failed: {e}")

        return issues

    def _c_cpp_static_analysis(self, code: str) -> List[Dict[str, Any]]:
        """C/C++ specific static analysis"""
        issues = []

        lines = code.split('\n')
        malloc_lines = []
        free_lines = []

        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # Track memory allocation
            if 'malloc(' in line_stripped or 'calloc(' in line_stripped:
                malloc_lines.append(i)

            if 'free(' in line_stripped:
                free_lines.append(i)

            # Check for dangerous functions
            if 'gets(' in line_stripped:
                issues.append({
                    'type': 'runtime_error',
                    'line': i,
                    'message': 'gets() is unsafe and deprecated',
                    'suggestions': ['Use fgets() instead']
                })

            # Check for buffer overflow potential
            if re.search(r'strcpy\s*\(', line_stripped):
                issues.append({
                    'type': 'runtime_error',
                    'line': i,
                    'message': 'strcpy may cause buffer overflow',
                    'suggestions': ['Use strncpy() or safer alternatives']
                })

            # Check for format string vulnerabilities
            if re.search(r'printf\s*\(\s*[^"\']*\w', line_stripped):
                issues.append({
                    'type': 'runtime_error',
                    'line': i,
                    'message': 'Potential format string vulnerability',
                    'suggestions': ['Use format strings properly']
                })

        # Check for memory leaks
        if len(malloc_lines) > len(free_lines):
            issues.append({
                'type': 'memory_error',
                'line': malloc_lines[0] if malloc_lines else 1,
                'message': 'Potential memory leak detected',
                'suggestions': ['Ensure all malloc/calloc calls have corresponding free calls']
            })

        return issues

    def _pattern_based_analysis(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Pattern-based analysis for common issues"""
        issues = []

        if language not in self.error_patterns:
            return issues

        patterns = self.error_patterns[language]
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            for pattern, message in patterns:
                if re.search(pattern, line):
                    issues.append({
                        'type': 'logic_error',
                        'line': i,
                        'message': message,
                        'suggestions': ['Review and fix the flagged pattern']
                    })

        return issues

    def _run_pylint(self, code: str) -> List[Dict[str, Any]]:
        """Run pylint on Python code"""
        issues = []

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file.flush()

                # Run pylint
                result = subprocess.run(
                    ['pylint', temp_file.name, '--output-format=json'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0 and result.stdout:
                    import json
                    try:
                        pylint_results = json.loads(result.stdout)
                        for item in pylint_results:
                            if item['type'] in ['error', 'warning']:
                                issues.append({
                                    'type': 'logic_error' if item['type'] == 'warning' else 'syntax_error',
                                    'line': item.get('line', 1),
                                    'message': f"Pylint {item['type']}: {item['message']}",
                                    'suggestions': ['Fix the pylint issue']
                                })
                    except (json.JSONDecodeError, KeyError):
                        pass

                # Clean up
                Path(temp_file.name).unlink()

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            self.logger.debug(f"Pylint analysis failed: {e}")

        return issues

    def extract_features(self, code: str, language: str) -> Dict[str, float]:
        """
        Extract numerical features from code for ML models

        Args:
            code: Source code as string
            language: Programming language

        Returns:
            Dictionary of numerical features
        """
        features = {}

        # Basic code metrics
        lines = code.split('\n')
        features['line_count'] = len(lines)
        features['char_count'] = len(code)
        features['avg_line_length'] = len(code) / len(lines) if lines else 0
        features['blank_line_ratio'] = sum(1 for line in lines if not line.strip()) / len(lines) if lines else 0

        # Comment analysis
        comment_chars = 0
        if language == 'python':
            comment_chars = sum(len(line.split('#', 1)[1]) for line in lines if '#' in line)
        elif language in ['c', 'cpp', 'javascript']:
            # Simple comment detection (doesn't handle /* */ properly)
            comment_chars = sum(len(line.split('//', 1)[1]) for line in lines if '//' in line)

        features['comment_ratio'] = comment_chars / len(code) if code else 0

        # Complexity metrics
        features['cyclomatic_complexity'] = self._calculate_complexity(code, language)
        features['nesting_depth'] = self._calculate_nesting_depth(code, language)
        features['function_count'] = self._count_functions(code, language)
        features['variable_count'] = self._count_variables(code, language)

        # Error-prone pattern counts
        features['dangerous_functions'] = self._count_dangerous_functions(code, language)
        features['magic_numbers'] = self._count_magic_numbers(code)
        features['long_lines'] = sum(1 for line in lines if len(line) > 80)

        # Language-specific features
        if language == 'python':
            features.update(self._python_features(code))
        elif language == 'javascript':
            features.update(self._javascript_features(code))
        elif language in ['c', 'cpp']:
            features.update(self._c_cpp_features(code))

        return features

    def _calculate_complexity(self, code: str, language: str) -> int:
        """Calculate cyclomatic complexity approximation"""
        complexity = 1  # Base complexity

        # Count decision points
        patterns = [
            r'\bif\b', r'\bwhile\b', r'\bfor\b', r'\bcase\b',
            r'\bcatch\b', r'\b\|\|\b', r'\b&&\b', r'\?\s*.*\s*:'
        ]

        for pattern in patterns:
            complexity += len(re.findall(pattern, code, re.IGNORECASE))

        return complexity

    def _calculate_nesting_depth(self, code: str, language: str) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0

        # Simple brace/indentation counting
        if language in ['c', 'cpp', 'javascript']:
            for char in code:
                if char == '{':
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == '}':
                    current_depth = max(0, current_depth - 1)

        elif language == 'python':
            lines = code.split('\n')
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    depth = indent // 4  # Assuming 4-space indentation
                    max_depth = max(max_depth, depth)

        return max_depth

    def _count_functions(self, code: str, language: str) -> int:
        """Count number of functions/methods"""
        if language == 'python':
            return len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))
        elif language == 'javascript':
            return len(re.findall(r'function\s+\w+|=>\s*\{|\w+\s*:\s*function', code))
        elif language in ['c', 'cpp']:
            return len(re.findall(r'^\s*\w+\s+\w+\s*\([^)]*\)\s*\{', code, re.MULTILINE))

        return 0

    def _count_variables(self, code: str, language: str) -> int:
        """Count number of variable declarations"""
        if language == 'python':
            return len(re.findall(r'^\s*\w+\s*=', code, re.MULTILINE))
        elif language == 'javascript':
            return len(re.findall(r'\b(var|let|const)\s+\w+', code))
        elif language in ['c', 'cpp']:
            return len(re.findall(r'\b(int|float|double|char|bool)\s+\w+', code))

        return 0

    def _count_dangerous_functions(self, code: str, language: str) -> int:
        """Count usage of dangerous/deprecated functions"""
        dangerous_funcs = {
            'python': ['eval', 'exec', 'input', 'raw_input'],
            'javascript': ['eval', 'setTimeout', 'setInterval'],
            'c': ['gets', 'strcpy', 'strcat', 'sprintf'],
            'cpp': ['gets', 'strcpy', 'strcat', 'sprintf']
        }

        if language not in dangerous_funcs:
            return 0

        count = 0
        for func in dangerous_funcs[language]:
            count += len(re.findall(rf'\b{func}\s*\(', code))

        return count

    def _count_magic_numbers(self, code: str) -> int:
        """Count magic numbers (hardcoded numeric literals)"""
        # Find numeric literals that aren't 0, 1, or in obvious contexts
        numbers = re.findall(r'\b(?!0\b|1\b)\d+(?:\.\d+)?\b', code)
        return len([n for n in numbers if float(n) not in [0, 1, 2, 10, 100]])

    def _python_features(self, code: str) -> Dict[str, float]:
        """Python-specific features"""
        features = {}

        # Import analysis
        features['import_count'] = len(re.findall(r'^\s*(import|from)\s+', code, re.MULTILINE))
        features['wildcard_imports'] = len(re.findall(r'from\s+\w+\s+import\s+\*', code))

        # Exception handling
        features['try_blocks'] = len(re.findall(r'^\s*try\s*:', code, re.MULTILINE))
        features['except_blocks'] = len(re.findall(r'^\s*except', code, re.MULTILINE))
        features['bare_excepts'] = len(re.findall(r'^\s*except\s*:', code, re.MULTILINE))

        # Class and method analysis
        features['class_count'] = len(re.findall(r'^\s*class\s+\w+', code, re.MULTILINE))
        features['method_count'] = len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))

        return features

    def _javascript_features(self, code: str) -> Dict[str, float]:
        """JavaScript-specific features"""
        features = {}

        # Variable declarations
        features['var_count'] = len(re.findall(r'\bvar\s+', code))
        features['let_count'] = len(re.findall(r'\blet\s+', code))
        features['const_count'] = len(re.findall(r'\bconst\s+', code))

        # Equality operators
        features['loose_equality'] = len(re.findall(r'==(?!=)', code))
        features['strict_equality'] = len(re.findall(r'===', code))

        # Event handlers
        features['event_listeners'] = len(re.findall(r'addEventListener|on\w+\s*=', code))

        return features

    def _c_cpp_features(self, code: str) -> Dict[str, float]:
        """C/C++ specific features"""
        features = {}

        # Memory management
        features['malloc_count'] = len(re.findall(r'\bmalloc\s*\(', code))
        features['free_count'] = len(re.findall(r'\bfree\s*\(', code))
        features['new_count'] = len(re.findall(r'\bnew\s+', code))
        features['delete_count'] = len(re.findall(r'\bdelete\s+', code))

        # Pointer usage
        features['pointer_declarations'] = len(re.findall(r'\w+\s*\*\s*\w+', code))
        features['pointer_dereferences'] = len(re.findall(r'\*\w+', code))

        # Include statements
        features['include_count'] = len(re.findall(r'#include', code))
        features['c_headers'] = len(re.findall(r'#include\s*<\w+\.h>', code))

        return features


# Example usage and testing
if __name__ == "__main__":
    preprocessor = CodePreprocessor()

    # Test Python code
    python_code = '''
def test_function():
    try:
        result = eval("2 + 2")
        print result  # Python 2 syntax
    except:
        pass
    return result
'''

    print("Python Analysis:")
    issues = preprocessor.static_analysis(python_code, 'python')
    for issue in issues:
        print(f"  {issue}")

    features = preprocessor.extract_features(python_code, 'python')
    print(f"  Features: {features}")