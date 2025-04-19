# DeepMed Regression Services

This directory contains microservices for regression modeling within the DeepMed application. These services provide functionality for data cleaning, feature selection, model training, and prediction specifically optimized for regression tasks.

## Services Overview

### Core Services

1. **Regression Data Cleaner (Port 5031)**
   - Cleans and preprocesses data specifically for regression tasks
   - LLM-powered optimization for regression datasets
   - Handles missing values, outliers, feature encoding, and transformations

2. **Regression Feature Selector (Port 5032)**
   - Selects features most relevant for regression tasks
   - LLM-powered analysis for important predictors
   - Performs aggressive feature reduction to improve model performance

3. **Regression Model Coordinator (Port 5040)**
   - Orchestrates the training of multiple regression models
   - Handles model selection and evaluation
   - Stores model metadata and artifacts

### Regression Model Services

1. **Linear Regression (Port 5041)**
   - Basic linear regression model
   - Optimized for medical regression tasks

2. **Lasso Regression (Port 5042)**
   - Linear regression with L1 regularization
   - Excellent for feature selection

3. **Ridge Regression (Port 5043)**
   - Linear regression with L2 regularization
   - Good for multicollinearity issues

4. **Random Forest Regression (Port 5044)**
   - Ensemble tree-based regression
   - Handles non-linearity well

5. **KNN Regression (Port 5045)**
   - K-Nearest Neighbors regression
   - Effective for certain pattern recognition tasks

6. **XGBoost Regression (Port 5046)**
   - Gradient boosted tree regression
   - High performance predictor

7. **Regression Predictor Service (Port 5050)**
   - Handles prediction requests for trained regression models

## Directory Structure

- `/regression_data_cleaner`: Data preprocessing service
- `/regression_feature_selector`: Feature selection service
- `/regression_model_coordinator`: Model orchestration service
- `/linear_regression`: Linear regression model service
- `/lasso_regression`: Lasso regression model service
- `/ridge_regression`: Ridge regression model service
- `/random_forest_regression`: Random forest regression model service
- `/knn_regression`: K-nearest neighbors regression model service
- `/xgboost_regression`: XGBoost regression model service
- `/regression_predictor`: Prediction service
- `/logs`: Log files
- `/saved_models`: Saved model artifacts
- `/mlruns`: MLflow tracking files

## Integration with DeepMed

These services integrate with the frontend application through the regression tab in the DeepMed interface. The workflow follows the same pattern as classification:

1. User uploads data through the Train Regression interface
2. Data is cleaned using the Regression Data Cleaner
3. Features are selected using the Regression Feature Selector
4. Multiple regression models are trained in parallel
5. The best models are selected and displayed to the user
6. Users can make predictions using the trained models

## Metrics

Regression models are evaluated using the following metrics:
- R² (Coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error) 