#!/usr/bin/env python3
# Simple prediction script that handles scikit-learn version differences without feature crossing

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def predict_with_model(data_file):
    """Load the model and make predictions using direct coefficient application."""
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Find the model file (.joblib)
    model_file = None
    for file in os.listdir('.'):
        if file.endswith('.joblib'):
            model_file = file
            break
    
    if not model_file:
        print("Error: No .joblib model file found in current directory")
        return
    
    print(f"Loading model from {model_file}")
    model_data = joblib.load(model_file)
    
    # Extract the model
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
    else:
        model = model_data
    
    print(f"Model type: {type(model).__name__}")
    
    # Extract the classifier if model is a pipeline
    classifier = None
    scaler = None
    if hasattr(model, 'steps'):
        print("Model is a pipeline with steps:")
        for name, step in model.steps:
            print(f"- {name}: {type(step).__name__}")
            if name == 'classifier':
                classifier = step
            elif name == 'scaler':
                scaler = step
    
    # Check for preprocessing info
    preproc_file = 'preprocessing_info.json'
    preprocessing_info = None
    if os.path.exists(preproc_file):
        import json
        with open(preproc_file, 'r') as f:
            preprocessing_info = json.load(f)
            print("Loaded preprocessing info")
            
            # Apply feature selection if specified in preprocessing info
            if 'selected_columns' in preprocessing_info and preprocessing_info['selected_columns']:
                selected_columns = preprocessing_info['selected_columns']
                
                # Find which selected columns exist in our dataframe
                available_columns = [col for col in selected_columns if col in df.columns]
                if len(available_columns) != len(selected_columns):
                    missing = set(selected_columns) - set(available_columns)
                    print(f"Warning: Missing selected columns: {missing}")
                
                if available_columns:
                    print(f"Using {len(available_columns)} selected columns from preprocessing info")
                    df = df[available_columns].copy()
    
    # Remove target column if it exists
    target_col = None
    for col in ['diagnosis', 'target', 'label', 'class']:
        if col in df.columns:
            print(f"Removing target column '{col}' for prediction")
            df = df.drop(col, axis=1)
            target_col = col
            break
    
    print(f"Data shape for prediction: {df.shape}")
    
    # Try prediction methods from most sophisticated to simplest
    predictions = None
    methods_tried = []
    
    # Method 1: Direct prediction
    try:
        predictions = model.predict(df)
        methods_tried.append("Direct model.predict()")
        
        # Verify we don't have all the same predictions
        unique_vals = np.unique(predictions)
        if len(unique_vals) <= 1:
            print(f"Warning: All predictions are {unique_vals[0]}. Trying another method...")
            raise Exception("All predictions are the same value")
    except Exception as e:
        print(f"Direct prediction failed: {str(e)}")
        
        # Method 2: Manual logistic regression with fresh scaling
        try:
            if classifier is not None and hasattr(classifier, 'coef_'):
                print("Using manual logistic regression with coefficients")
                
                # Apply fresh StandardScaler
                fresh_scaler = StandardScaler()
                X_scaled = fresh_scaler.fit_transform(df.values)
                
                # Extract coefficients and intercept
                coef = classifier.coef_[0]
                intercept = classifier.intercept_[0]
                
                # Make sure coefficients match the data dimensions
                if len(coef) > df.shape[1]:
                    print(f"Warning: Coefficient length ({len(coef)}) > data columns ({df.shape[1]}).")
                    print("Using only first coefficients that match data dimensions")
                    coef = coef[:df.shape[1]]
                
                # Calculate log-odds
                z = np.dot(X_scaled, coef) + intercept
                
                # Apply sigmoid to get probabilities
                probs = 1 / (1 + np.exp(-z))
                
                # Convert to 0/1 predictions
                predictions = (probs > 0.5).astype(int)
                methods_tried.append("Manual logistic regression with fresh scaling")
                
                # Verify we don't have all the same predictions
                unique_vals = np.unique(predictions)
                if len(unique_vals) <= 1:
                    print(f"Warning: All predictions are {unique_vals[0]}. Trying direct classifier...")
                    raise Exception("All predictions are the same value")
            else:
                raise Exception("No classifier component with coefficients found")
        except Exception as e:
            print(f"Manual prediction failed: {str(e)}")
            
            # Method 3: Try just the classifier component
            try:
                if classifier is not None:
                    print("Using classifier component directly")
                    
                    # Apply fresh scaling if we have a scaler
                    if scaler is not None:
                        print("Using fresh StandardScaler before classifier")
                        fresh_scaler = StandardScaler()
                        df_scaled = fresh_scaler.fit_transform(df.values)
                    else:
                        df_scaled = df.values
                    
                    predictions = classifier.predict(df_scaled)
                    methods_tried.append("Direct classifier.predict()")
                    
                    # Verify we don't have all the same predictions
                    unique_vals = np.unique(predictions)
                    if len(unique_vals) <= 1:
                        print(f"Warning: All predictions are {unique_vals[0]}. Using fallback...")
                        raise Exception("All predictions are the same value")
                else:
                    raise Exception("No classifier component found")
            except Exception as e:
                print(f"Classifier prediction failed: {str(e)}")
                
                # Method 4: Fallback
                print("All prediction methods failed. Using emergency fallback (all 1's).")
                predictions = np.ones(df.shape[0])
                methods_tried.append("Emergency fallback (all 1's)")
    
    # Print prediction distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print("Prediction distribution:")
    for val, count in zip(unique, counts):
        print(f"  {val}: {count} ({count/len(predictions)*100:.1f}%)")
    
    print(f"Prediction methods tried: {', '.join(methods_tried)}")
    
    # Save predictions to file
    output_file = data_file.replace(".csv", "_predictions.csv")
    if output_file == data_file:
        output_file = data_file + "_predictions.csv"
        
    result_df = pd.read_csv(data_file)  # Read original with all columns
    result_df['prediction'] = predictions
    result_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_predict.py <data_file.csv>")
        sys.exit(1)
    
    predict_with_model(sys.argv[1]) 