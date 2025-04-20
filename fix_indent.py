#!/usr/bin/env python3


def fix_file():
    file_path = "quangtps/dose/dose_calculator.py"
    fixed_path = "quangtps/dose/dose_calculator_fixed.py"

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        content = file.readlines()

    # Check lines around 545-550
    for i in range(540, 555):
        print(f"Line {i + 1}: {repr(content[i])}")

    # Fix the indentation issue
    # Make sure there's a blank line between the return statement and the next method
    if "return True" in content[547] and not content[548].strip():
        print("The file already has correct indentation")
    else:
        # Ensure there's a blank line after return True
        content[547] = "        return True\n"
        content.insert(548, "\n")
        print("Fixed indentation issue")

    # Check lines around 809
    print("\nChecking lines around 809:")
    for i in range(804, 814):
        print(f"Line {i + 1}: {repr(content[i])}")

    # Fix indentation issues around line 809
    # Determine the correct indentation by looking at surrounding lines
    correct_indent = None
    for i in range(805, 813):
        line = content[i].rstrip()
        if line and not line.isspace():
            # Get the indentation of this line
            indent = len(line) - len(line.lstrip())
            if line.lstrip().startswith("def "):
                # This is a method definition, should have 4 spaces
                correct_indent = " " * 4
                break
            elif indent > 0:
                # Use the indentation of this non-empty line
                correct_indent = " " * indent
                break

    if correct_indent:
        # Fix the indentation of line 809
        line_content = content[809].lstrip()
        content[809] = correct_indent + line_content
        print(f"Fixed indentation at line 810: {repr(content[809])}")

    # Write the fixed content
    with open(fixed_path, "w", encoding="utf-8") as file:
        file.writelines(content)

    print(f"Fixed file written to {fixed_path}")


if __name__ == "__main__":
    fix_file()
