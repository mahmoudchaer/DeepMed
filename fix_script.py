#!/usr/bin/env python3
# Script to fix the app_api.py file

with open('app_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if the script_content has proper closing triple quotes
if "print(f\"Error during prediction: {str(e)}\")\ndef create_readme_file" in content:
    fixed_content = content.replace(
        "print(f\"Error during prediction: {str(e)}\")\ndef create_readme_file",
        "print(f\"Error during prediction: {str(e)}\")\n'''\n\n    # Write the utility script to a file\n    script_path = os.path.join(temp_dir, 'predict.py')\n    with open(script_path, 'w') as f:\n        f.write(script_content)\n\ndef create_readme_file"
    )
    
    # Save the fixed content
    with open('app_api.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print("Fixed app_api.py successfully!")
else:
    print("No fix needed or pattern not found.") 