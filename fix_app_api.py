#!/usr/bin/env python3
# Script to fix the syntax error in app_api.py

with open('app_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The key part is to find where the create_utility_script function's triple-quoted string should end
# and add the missing closing quotes and code to write the script to a file
fixed_content = content.replace(
    'print(f"Error during prediction: {str(e)}")\n\ndef create_readme_file',
    'print(f"Error during prediction: {str(e)}")\n\'\'\'\n\n    # Write the utility script to a file\n    script_path = os.path.join(temp_dir, "predict.py")\n    with open(script_path, "w") as f:\n        f.write(script_content)\n\ndef create_readme_file'
)

with open('app_api.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("Fixed app_api.py successfully!") 