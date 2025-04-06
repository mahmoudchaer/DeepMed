#!/usr/bin/env python3
# Direct fix for the syntax error in app_api.py

with open('app_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Look for the last line of the script_content = ''' block that appears truncated
if 'mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)' in content:
    # Fix the truncated code section by finding the text and adding proper ending
    fixed_content = content.replace(
        'mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)',
        'mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)'
    )
    
    # Now add the missing triple quotes and function code right before def create_readme_file
    if 'def create_readme_file' in fixed_content:
        fixed_content = fixed_content.replace(
            'def create_readme_file',
            '''                if self.classifier is not None:
                    # This is the logistic regression part if available
                    print("Using coefficient/intercept method for prediction")
                    # Apply StandardScaler ourselves since we might be using a different sklearn version
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(processed_df)
                    
                    # Get coefficients and intercept
                    coefficients = self.classifier.coef_[0] if hasattr(self.classifier, 'coef_') else None
                    intercept = self.classifier.intercept_[0] if hasattr(self.classifier, 'intercept_') else 0
                    
                    if coefficients is not None:
                        # Manual prediction using the logistic function
                        z = np.dot(scaled_data, coefficients) + intercept
                        predictions = 1 / (1 + np.exp(-z))
                        predictions = (predictions > 0.5).astype(int)  # Convert to binary prediction
                        print("Applied logistic regression formula with coefficients")
                    else:
                        # Fallback to direct prediction on classifier only
                        predictions = self.classifier.predict(processed_df)
                        print("Used classifier component directly")
                else:
                    raise Exception("No classifier component found in model")
                
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
'''
+ "'''\n\n    # Write the utility script to a file\n    script_path = os.path.join(temp_dir, 'predict.py')\n    with open(script_path, 'w') as f:\n        f.write(script_content)\n\ndef create_readme_file"
        )
    
    # Save the fixed content
    with open('app_api.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Fixed app_api.py successfully!")
else:
    print("Could not locate the specific code section to fix. Manual intervention may be needed.") 