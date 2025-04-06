#!/usr/bin/env python3
# Direct fix for unterminated triple-quoted string in app_api.py

# Read the file
with open('app_api.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where the triple quote starts (around line 2402)
start_line = 0
for i, line in enumerate(lines):
    if 'script_content = \'\'\'' in line:
        start_line = i
        break

# Find the end of the create_utility_script function (before create_readme_file)
end_function_line = 0
for i, line in enumerate(lines[start_line:], start_line):
    if 'def create_readme_file' in line:
        end_function_line = i
        break

# Insert the closing triple quotes and file writing code before create_readme_file
if start_line > 0 and end_function_line > start_line:
    # Add closing triple quotes and code to write the script to a file
    lines.insert(end_function_line, "    # Write the utility script to a file\n    script_path = os.path.join(temp_dir, 'predict.py')\n    with open(script_path, 'w') as f:\n        f.write(script_content)\n\n")
    lines.insert(end_function_line, "'''\n\n")
    
    # Write the fixed content back to the file
    with open('app_api.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Fixed app_api.py successfully! Added closing triple quotes at line {end_function_line}")
else:
    print("Could not locate the proper lines to fix. Please check the file manually.") 