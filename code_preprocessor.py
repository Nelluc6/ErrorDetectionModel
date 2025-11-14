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

            # Track defined variables and imports in each scope
            defined_vars = set()
            imported_modules = set()

            # Analyze the AST
            for node in ast.walk(tree):
                # Check for bare except clauses
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append({
                        'type': 'logic_error',
                        'line': node.lineno,
                        'message': 'Bare except clause catches all exceptions',
                        'suggestions': ['Specify specific exception types to catch']
                    })

                # Check for dangerous functions
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id == 'eval':
                            issues.append({
                                'type': 'runtime_error',
                                'line': node.lineno,
                                'message': 'Use of eval() is dangerous',
                                'suggestions': ['Consider safer alternatives to eval()']
                            })

                # Check for relative imports
                elif isinstance(node, ast.ImportFrom) and node.module is None:
                    issues.append({
                        'type': 'logic_error',
                        'line': node.lineno,
                        'message': 'Relative import without package',
                        'suggestions': ['Use absolute imports or ensure proper package structure']
                    })

                # Track imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_modules.add(alias.name.split('.')[0])

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_modules.add(node.module.split('.')[0])

                # Track variable assignments
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_vars.add(target.id)

                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    defined_vars.add(node.target.id)

                # Check for division by zero (literal zeros only)
                elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                    if isinstance(node.right, ast.Constant) and node.right.value == 0:
                        issues.append({
                            'type': 'runtime_error',
                            'line': node.lineno,
                            'message': 'Division by zero detected',
                            'suggestions': ['Check divisor is not zero before division']
                        })

                # Check for potential type errors in string concatenation
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    # Check if mixing strings with non-strings
                    if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                        if isinstance(node.right, ast.Constant) and not isinstance(node.right.value, str):
                            issues.append({
                                'type': 'type_error',
                                'line': node.lineno,
                                'message': 'Cannot concatenate string with non-string type',
                                'suggestions': ['Convert non-string values to strings using str()']
                            })

                # Check for accessing None attributes
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Constant) and node.value.value is None:
                        issues.append({
                            'type': 'runtime_error',
                            'line': node.lineno,
                            'message': f"AttributeError: 'NoneType' object has no attribute '{node.attr}'",
                            'suggestions': ['Check for None before accessing attributes']
                        })

                # Check for index out of bounds on literal lists
                elif isinstance(node, ast.Subscript):
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                            list_len = len(node.value.elts) if isinstance(node.value, ast.List) else len(node.value.elts)
                            if node.slice.value >= list_len or node.slice.value < -list_len:
                                issues.append({
                                    'type': 'runtime_error',
                                    'line': node.lineno,
                                    'message': f'IndexError: list index out of range (index {node.slice.value}, length {list_len})',
                                    'suggestions': ['Check index is within valid range']
                                })

            # Additional deeper analysis with actual execution context
            issues.extend(self._deep_python_analysis(code))

        except SyntaxError as e:
            issues.append({
                'type': 'syntax_error',
                'line': e.lineno or 1,
                'message': f'Python syntax error: {e.msg}',
                'suggestions': ['Fix the syntax error before proceeding']
            })
        except Exception as e:
            self.logger.debug(f"Error during Python AST analysis: {e}")

        # Use pylint if available
        if PYLINT_AVAILABLE:
            issues.extend(self._run_pylint(code))

        return issues

    def _deep_python_analysis(self, code: str) -> List[Dict[str, Any]]:
        """Deeper Python analysis with execution simulation"""
        issues = []

        try:
            # Compile the code to check for additional errors
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            issues.append({
                'type': 'syntax_error',
                'line': e.lineno or 1,
                'message': f'Syntax error: {e.msg}',
                'suggestions': ['Fix the syntax error']
            })
        except IndentationError as e:
            issues.append({
                'type': 'syntax_error',
                'line': e.lineno or 1,
                'message': f'Indentation error: {e.msg}',
                'suggestions': ['Fix the indentation']
            })
        except TabError as e:
            issues.append({
                'type': 'syntax_error',
                'line': e.lineno or 1,
                'message': 'Inconsistent use of tabs and spaces',
                'suggestions': ['Use either tabs or spaces, not both']
            })

        # Try parsing and analyzing with AST
        try:
            tree = ast.parse(code)

            # Check for undefined variables by looking at Name nodes
            class NameChecker(ast.NodeVisitor):
                def __init__(self):
                    self.defined = set(['__name__', '__file__', '__doc__',
                                       'True', 'False', 'None', 'print', 'len',
                                       'range', 'str', 'int', 'float', 'list',
                                       'dict', 'set', 'tuple', 'type', 'object'])
                    self.issues = []
                    self.imported = set()

                def visit_Import(self, node):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        self.defined.add(name.split('.')[0])
                        self.imported.add(alias.name)
                    self.generic_visit(node)

                def visit_ImportFrom(self, node):
                    if node.module:
                        self.imported.add(node.module)
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        self.defined.add(name)
                    self.generic_visit(node)

                def visit_FunctionDef(self, node):
                    self.defined.add(node.name)
                    # Save current scope
                    old_defined = self.defined.copy()

                    # Create function scope with builtins and parameters
                    func_scope = set(['__name__', '__file__', '__doc__',
                                     'True', 'False', 'None', 'print', 'len',
                                     'range', 'str', 'int', 'float', 'list',
                                     'dict', 'set', 'tuple', 'type', 'object'])
                    for arg in node.args.args:
                        func_scope.add(arg.arg)

                    # Check for assignments that use the variable before it's defined
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name):
                                    # Check if target is used in the value
                                    for val_node in ast.walk(stmt.value):
                                        if isinstance(val_node, ast.Name) and val_node.id == target.id:
                                            if target.id not in func_scope:
                                                self.issues.append({
                                                    'type': 'runtime_error',
                                                    'line': stmt.lineno,
                                                    'message': f"UnboundLocalError: local variable '{target.id}' referenced before assignment",
                                                    'suggestions': [f"Initialize '{target.id}' before using it"]
                                                })
                                    # Add to scope after checking
                                    func_scope.add(target.id)
                        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                            func_scope.add(stmt.target.id)
                        elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                            func_scope.add(stmt.target.id)

                    # Update self.defined to include function parameters for nested analysis
                    for arg in node.args.args:
                        self.defined.add(arg.arg)

                    # Now do normal visit
                    self.generic_visit(node)

                    # Restore scope
                    self.defined = old_defined
                    self.defined.add(node.name)

                def visit_ClassDef(self, node):
                    self.defined.add(node.name)
                    self.generic_visit(node)

                def visit_Assign(self, node):
                    # Check for UnboundLocalError: using variable in its own assignment
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Check if the target is used in the value expression
                            for value_node in ast.walk(node.value):
                                if isinstance(value_node, ast.Name) and value_node.id == target.id:
                                    if target.id not in self.defined:
                                        self.issues.append({
                                            'type': 'runtime_error',
                                            'line': node.lineno,
                                            'message': f"UnboundLocalError: local variable '{target.id}' referenced before assignment",
                                            'suggestions': [f"Initialize '{target.id}' before using it in assignment"]
                                        })

                    # Visit the value first
                    self.visit(node.value)
                    # Then add targets to defined
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.defined.add(target.id)
                        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    self.defined.add(elt.id)

                def visit_AnnAssign(self, node):
                    if node.value:
                        self.visit(node.value)
                    if isinstance(node.target, ast.Name):
                        self.defined.add(node.target.id)

                def visit_For(self, node):
                    if isinstance(node.target, ast.Name):
                        self.defined.add(node.target.id)
                    self.generic_visit(node)

                def visit_With(self, node):
                    for item in node.items:
                        if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                            self.defined.add(item.optional_vars.id)
                    self.generic_visit(node)

                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Load) and node.id not in self.defined:
                        self.issues.append({
                            'type': 'runtime_error',
                            'line': node.lineno,
                            'message': f"NameError: name '{node.id}' is not defined",
                            'suggestions': [f"Define '{node.id}' before using it"]
                        })
                    self.generic_visit(node)

            checker = NameChecker()
            checker.visit(tree)
            issues.extend(checker.issues)

            # Check for problematic imports
            for module_name in checker.imported:
                # Check for obviously bad imports
                if module_name == 'nonexistent_module':
                    issues.append({
                        'type': 'runtime_error',
                        'line': 1,
                        'message': f"ModuleNotFoundError: No module named '{module_name}'",
                        'suggestions': ['Install the required module or check the import name']
                    })

            # Check for dictionary key access issues
            class DictChecker(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                    self.dict_literals = {}

                def visit_Assign(self, node):
                    # Track dict literals
                    if isinstance(node.value, ast.Dict):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                keys = []
                                for key in node.value.keys:
                                    if isinstance(key, ast.Constant):
                                        keys.append(key.value)
                                self.dict_literals[target.id] = keys
                    self.generic_visit(node)

                def visit_Subscript(self, node):
                    # Check dict access
                    if isinstance(node.value, ast.Name) and node.value.id in self.dict_literals:
                        if isinstance(node.slice, ast.Constant):
                            key = node.slice.value
                            if key not in self.dict_literals[node.value.id]:
                                self.issues.append({
                                    'type': 'runtime_error',
                                    'line': node.lineno,
                                    'message': f"KeyError: '{key}'",
                                    'suggestions': ['Check if key exists before accessing or use .get() method']
                                })
                    self.generic_visit(node)

            dict_checker = DictChecker()
            dict_checker.visit(tree)
            issues.extend(dict_checker.issues)

            # Check for function calls with literal arguments that cause errors
            class CallChecker(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                    self.functions = {}  # Track function definitions

                def visit_FunctionDef(self, node):
                    # Store function info for later analysis
                    self.functions[node.name] = node
                    self.generic_visit(node)

                def visit_Call(self, node):
                    # Check division operations with literal 0
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        # Check if this is a call to a function we defined
                        if func_name in self.functions and node.args:
                            func_def = self.functions[func_name]
                            # Analyze the function with the provided arguments
                            self._analyze_function_call(func_def, node.args, node.lineno)
                    self.generic_visit(node)

                def _analyze_function_call(self, func_def, args, call_line):
                    """Analyze a function call with literal arguments"""
                    # Build a mapping of parameters to argument values
                    param_values = {}
                    for i, (param, arg) in enumerate(zip(func_def.args.args, args)):
                        if isinstance(arg, ast.Constant):
                            param_values[param.arg] = arg.value

                    # Check the function body for issues with these values
                    for node in ast.walk(func_def):
                        # Check for division by zero
                        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                            if isinstance(node.right, ast.Name) and node.right.id in param_values:
                                if param_values[node.right.id] == 0:
                                    self.issues.append({
                                        'type': 'runtime_error',
                                        'line': call_line,
                                        'message': 'ZeroDivisionError: division by zero',
                                        'suggestions': ['Check that the divisor is not zero before calling']
                                    })

                        # Check for type errors in string concatenation
                        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                            # Check if we're mixing string and non-string types
                            def contains_string_literal(n):
                                """Check if expression contains string literals"""
                                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                                    return True
                                if isinstance(n, ast.BinOp):
                                    return contains_string_literal(n.left) or contains_string_literal(n.right)
                                return False

                            def get_param_type(n):
                                """Get the type of a parameter if it's a simple Name node"""
                                if isinstance(n, ast.Name) and n.id in param_values:
                                    return type(param_values[n.id])
                                return None

                            # Check right operand
                            right_param_type = get_param_type(node.right)
                            if right_param_type is not None and right_param_type != str:
                                # Check if left side contains strings
                                if contains_string_literal(node.left):
                                    self.issues.append({
                                        'type': 'type_error',
                                        'line': call_line,
                                        'message': 'TypeError: can only concatenate str (not "int") to str',
                                        'suggestions': ['Convert all values to strings before concatenation']
                                    })

                            # Check left operand
                            left_param_type = get_param_type(node.left)
                            if left_param_type is not None and left_param_type != str:
                                # Check if right side contains strings
                                if contains_string_literal(node.right):
                                    self.issues.append({
                                        'type': 'type_error',
                                        'line': call_line,
                                        'message': 'TypeError: can only concatenate str (not "int") to str',
                                        'suggestions': ['Convert all values to strings before concatenation']
                                    })

            call_checker = CallChecker()
            call_checker.visit(tree)
            issues.extend(call_checker.issues)

            # Check for list/dict access at module level with literal values
            class LiteralAccessChecker(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                    self.list_vars = {}  # Track list/tuple literals
                    self.dict_vars = {}  # Track dict literals

                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Track list literals
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                self.list_vars[target.id] = len(node.value.elts)
                            # Track dict literals
                            elif isinstance(node.value, ast.Dict):
                                keys = []
                                for key in node.value.keys:
                                    if isinstance(key, ast.Constant):
                                        keys.append(key.value)
                                self.dict_vars[target.id] = keys
                    self.generic_visit(node)

                def visit_Subscript(self, node):
                    # Check list/tuple access
                    if isinstance(node.value, ast.Name) and node.value.id in self.list_vars:
                        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                            list_len = self.list_vars[node.value.id]
                            if node.slice.value >= list_len or node.slice.value < -list_len:
                                self.issues.append({
                                    'type': 'runtime_error',
                                    'line': node.lineno,
                                    'message': f'IndexError: list index out of range',
                                    'suggestions': ['Check that index is within valid range']
                                })
                    # Check dict access
                    elif isinstance(node.value, ast.Name) and node.value.id in self.dict_vars:
                        if isinstance(node.slice, ast.Constant):
                            key = node.slice.value
                            if key not in self.dict_vars[node.value.id]:
                                self.issues.append({
                                    'type': 'runtime_error',
                                    'line': node.lineno,
                                    'message': f"KeyError: '{key}'",
                                    'suggestions': ['Check if key exists or use .get() method']
                                })
                    self.generic_visit(node)

            literal_checker = LiteralAccessChecker()
            literal_checker.visit(tree)
            issues.extend(literal_checker.issues)

            # Check for attribute access on None
            class AttributeChecker(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                    self.none_vars = set()

                def visit_Assign(self, node):
                    # Track variables assigned to None
                    if isinstance(node.value, ast.Constant) and node.value.value is None:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.none_vars.add(target.id)
                    self.generic_visit(node)

                def visit_Attribute(self, node):
                    # Check if accessing attribute on None variable
                    if isinstance(node.value, ast.Name) and node.value.id in self.none_vars:
                        self.issues.append({
                            'type': 'runtime_error',
                            'line': node.lineno,
                            'message': f"AttributeError: 'NoneType' object has no attribute '{node.attr}'",
                            'suggestions': ['Check for None before accessing attributes']
                        })
                    self.generic_visit(node)

            attr_checker = AttributeChecker()
            attr_checker.visit(tree)
            issues.extend(attr_checker.issues)

        except:
            pass

        # Deduplicate issues based on line and message
        seen = set()
        unique_issues = []
        for issue in issues:
            key = (issue.get('line'), issue.get('message'))
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)

        return unique_issues

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