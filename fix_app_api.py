#!/usr/bin/env python3
# Script to fix the app_api.py file

with open('app_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the line with script_content = ''' (around line 2402)
# and add proper closing triple quotes if missing
if 'script_content = \'\'\'' in content and '\'\'\'\\n\\n    # Write the utility script to a file' not in content:
    fixed_content = content.replace(
        'script_content = \'\'\'',
        'script_content = \'\'\'',
        1  # Replace only the first occurrence to preserve other instances
    )
    
    # Find where to add the closing quotes - before create_readme_file function
    if 'def create_readme_file' in fixed_content:
        fixed_content = fixed_content.replace(
            'def create_readme_file',
            '\'\'\'\\n\\n    # Write the utility script to a file\\n    script_path = os.path.join(temp_dir, \'predict.py\')\\n    with open(script_path, \'w\') as f:\\n        f.write(script_content)\\n\\ndef create_readme_file',
            1  # Replace only the first occurrence
        )
        
    # Save the fixed content
    with open('app_api.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print("Fixed app_api.py successfully!")
else:
    print("No fix needed or pattern not found.") 