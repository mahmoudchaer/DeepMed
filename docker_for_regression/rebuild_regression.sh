#!/bin/bash

# Stop any running regression services
echo "Stopping existing regression services..."
docker-compose down

# Remove previous images to ensure a clean rebuild
echo "Removing previous regression images..."
docker rmi -f docker_for_regression_regression-model-coordinator
docker rmi -f docker_for_regression_linear_regression
docker rmi -f docker_for_regression_lasso_regression
docker rmi -f docker_for_regression_ridge_regression
docker rmi -f docker_for_regression_random_forest_regression
docker rmi -f docker_for_regression_knn_regression
docker rmi -f docker_for_regression_xgboost_regression
docker rmi -f docker_for_regression_regression-predictor-service
docker rmi -f docker_for_regression_regression-data-cleaner
docker rmi -f docker_for_regression_regression-feature-selector

# Build the services
echo "Building regression services..."
docker-compose build

# Start services
echo "Starting regression services..."
docker-compose up -d

# Monitor the logs of the regression model coordinator service
echo "Monitoring logs of regression model coordinator..."
docker-compose logs -f regression-model-coordinator 