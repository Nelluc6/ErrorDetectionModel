#!/usr/bin/env python3
"""
Sample Python file with potential errors for testing the GUI

This file contains various types of errors that the error detection model
should be able to identify.
"""

def divide_numbers(a, b):
    """Function with potential division by zero error"""
    return a / b  # Potential ZeroDivisionError

def access_list_item(items, index):
    """Function with potential index error"""
    return items[index]  # Potential IndexError

def undefined_variable_usage():
    """Function with undefined variable usage"""
    print(some_undefined_variable)  # NameError

def type_error_example():
    """Function with potential type error"""
    return "string" + 5  # TypeError

def main():
    """Main function with various error scenarios"""
    # Division by zero
    result1 = divide_numbers(10, 0)

    # Index out of range
    my_list = [1, 2, 3]
    result2 = access_list_item(my_list, 10)

    # Undefined variable
    undefined_variable_usage()

    # Type error
    result3 = type_error_example()

    print("All operations completed")

if __name__ == "__main__":
    main()