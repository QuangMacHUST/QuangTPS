#!/usr/bin/env python3

import ast
import tokenize
import io
import traceback


def check_file_syntax(filename):
    """
    Check the syntax of a Python file and report errors.
    """
    with open(filename, "r", encoding="utf-8", errors="ignore") as file:
        content = file.read()

    print(f"Checking syntax of {filename}")

    # Try to parse the file with ast
    try:
        ast.parse(content, filename=filename)
        print("No syntax errors detected by AST parser.")
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        print(f"Line {e.lineno}, column {e.offset}: {e.text}")

        # Print context around the error
        lines = content.splitlines()
        start_line = max(0, e.lineno - 5)
        end_line = min(len(lines), e.lineno + 5)

        print("\nContext:")
        for i in range(start_line, end_line):
            prefix = ">" if i + 1 == e.lineno else " "
            print(f"{prefix} {i + 1:4d}: {lines[i]}")

    # Try to tokenize the file
    try:
        with open(filename, "rb") as file:
            tokens = list(tokenize.tokenize(file.readline))
        print("Successfully tokenized the file.")
    except tokenize.TokenError as e:
        print(f"Tokenization error: {e}")

    # Focus on lines 809-810
    print("\nChecking specifically lines 809-810:")
    lines = content.splitlines()
    if len(lines) >= 810:
        print(f"Line 809: {repr(lines[808])}")
        print(f"Line 810: {repr(lines[809])}")
        # Check indentation
        if lines[808].startswith(" "):
            spaces = len(lines[808]) - len(lines[808].lstrip())
            print(f"Line 809 has {spaces} leading spaces")
        else:
            print("Line 809 has no leading spaces")

        if lines[809].startswith(" "):
            spaces = len(lines[809]) - len(lines[809].lstrip())
            print(f"Line 810 has {spaces} leading spaces")
        else:
            print("Line 810 has no leading spaces")

        # Check surrounding lines
        print("\nSurrounding lines:")
        for i in range(max(0, 809 - 5), min(len(lines), 809 + 5)):
            print(f"{i + 1:4d}: {repr(lines[i])}")
            if lines[i].strip() and not lines[i].isspace():
                spaces = len(lines[i]) - len(lines[i].lstrip())
                print(f"      (indentation: {spaces} spaces)")


def fix_indentation(filename, outfilename):
    """
    Fix indentation issues in a Python file.
    """
    with open(filename, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    # Fix line 809
    if len(lines) >= 809:
        # Determine the correct indentation level from context
        # Look at the previous def line to find the method indentation level
        method_indent = 4  # Default method indentation (4 spaces)
        body_indent = 8  # Default method body indentation (8 spaces)

        for i in range(808, max(0, 808 - 50), -1):
            line = lines[i].rstrip()
            if line.lstrip().startswith("def "):
                # Found the method definition
                method_indent = len(line) - len(line.lstrip())
                body_indent = method_indent + 4
                break

        # Fix line 809
        line_content = lines[808].lstrip()
        lines[808] = " " * body_indent + line_content

        print(f"Fixed line 809: {repr(lines[808])}")

        # Fix line 810 if it's a method definition
        if len(lines) >= 810 and "def " in lines[809]:
            # This is likely a new method definition, should be at method_indent level
            line_content = lines[809].lstrip()
            lines[809] = " " * method_indent + line_content
            print(f"Fixed line 810: {repr(lines[809])}")

    # Write the fixed file
    with open(outfilename, "w", encoding="utf-8") as file:
        file.writelines(lines)

    print(f"Written fixed content to {outfilename}")


if __name__ == "__main__":
    filename = "quangtps/dose/dose_calculator.py"
    outfilename = "quangtps/dose/dose_calculator_fixed.py"

    check_file_syntax(filename)
    fix_indentation(filename, outfilename)
