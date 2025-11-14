"""
bad_examples.py - File with multiple errors for GUI testing
This file contains various Python errors that should be detected during analysis.
"""

import sys
import os

# Error 1: Division by zero
def calculate_average(total, count):
    return total / count  # Will fail if count is 0

result = calculate_average(100, 0)  # ZeroDivisionError

# Error 2: Undefined variable
def process_data():
    x = 10
    y = 20
    z=5
    return x + y + z  # NameError: 'z' is not defined

# Error 3: Type error - concatenating incompatible types
def format_message(name, age):
    return "User " + name + " is " + age + " years old"  # TypeError: can only concatenate str

message = format_message("Alice", 25)

# Error 4: Index out of bounds
data = [1, 2, 3, 4, 5]
print(data[10])  # IndexError

# Error 5: Key error in dictionary
user_data = {"name": "Bob", "age": 30}
email = user_data["email"]  # KeyError

# Error 6: Attribute error
value = None
length = value.strip()  # AttributeError: 'NoneType' object has no attribute 'strip'

# Error 7: Import error (if module doesn't exist)
import nonexistent_module  # ModuleNotFoundError

# Error 8: Incorrect indentation (syntax error)
def broken_function():
    if True:
        print("This is incorrectly indented")  # IndentationError

# Error 9: Using variable before assignment
def compute():
    result = result + 1  # UnboundLocalError
    return result

# Error 10: Invalid syntax - missing colon
if True:
    print("Missing colon")  # SyntaxError
