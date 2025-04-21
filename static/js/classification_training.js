document.addEventListener('DOMContentLoaded', function() {
    // Initialize the training overlay
    initTrainingOverlay();
    
    // Handle form submission for the training form
    const trainingForm = document.getElementById('training-form');
    if (trainingForm) {
        trainingForm.addEventListener('submit', function(e) {
            // Show the loading overlay
            showTrainingOverlay();
        });
    }
});

/**
 * Initialize the training overlay by creating and appending it to the document body
 */
function initTrainingOverlay() {
    // Create the overlay container
    const overlay = document.createElement('div');
    overlay.id = 'classification-loading-overlay';
    overlay.className = 'classification-loading-overlay';
    
    // Create overlay content with model cards
    overlay.innerHTML = `
        <div class="overlay-content">
            <div class="title-container">
                <h2><i class="fas fa-brain mr-2"></i> Training Classification Models</h2>
                <p>Our AI is building multiple models to find the best one for your data</p>
            </div>
            
            <div class="main-progress-container">
                <h5>Overall Progress</h5>
                <div class="progress main-progress">
                    <div id="main-progress-bar" class="progress-bar progress-bar-striped progress-bar-animated bg-primary" 
                         role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>
                </div>
                <p id="main-status-text" class="mt-2">Initializing training pipeline...</p>
            </div>
            
            <div class="models-container">
                <!-- Logistic Regression -->
                <div class="model-card in-progress fade-in" id="model-card-logistic-regression">
                    <div class="model-card-header">
                        <div class="model-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <h3 class="model-name">Logistic Regression</h3>
                    </div>
                    <div class="model-status" id="status-logistic-regression">Initializing...</div>
                    <div class="model-progress">
                        <div class="progress">
                            <div id="progress-logistic-regression" class="progress-bar progress-bar-striped progress-bar-animated bg-warning" 
                                 role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Decision Tree -->
                <div class="model-card in-progress fade-in" id="model-card-decision-tree">
                    <div class="model-card-header">
                        <div class="model-icon">
                            <i class="fas fa-sitemap"></i>
                        </div>
                        <h3 class="model-name">Decision Tree</h3>
                    </div>
                    <div class="model-status" id="status-decision-tree">Initializing...</div>
                    <div class="model-progress">
                        <div class="progress">
                            <div id="progress-decision-tree" class="progress-bar progress-bar-striped progress-bar-animated bg-warning" 
                                 role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Random Forest -->
                <div class="model-card in-progress fade-in" id="model-card-random-forest">
                    <div class="model-card-header">
                        <div class="model-icon">
                            <i class="fas fa-tree"></i>
                        </div>
                        <h3 class="model-name">Random Forest</h3>
                    </div>
                    <div class="model-status" id="status-random-forest">Initializing...</div>
                    <div class="model-progress">
                        <div class="progress">
                            <div id="progress-random-forest" class="progress-bar progress-bar-striped progress-bar-animated bg-warning" 
                                 role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- KNN -->
                <div class="model-card in-progress fade-in" id="model-card-knn">
                    <div class="model-card-header">
                        <div class="model-icon">
                            <i class="fas fa-project-diagram"></i>
                        </div>
                        <h3 class="model-name">K-Nearest Neighbors</h3>
                    </div>
                    <div class="model-status" id="status-knn">Initializing...</div>
                    <div class="model-progress">
                        <div class="progress">
                            <div id="progress-knn" class="progress-bar progress-bar-striped progress-bar-animated bg-warning" 
                                 role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- SVM -->
                <div class="model-card in-progress fade-in" id="model-card-svm">
                    <div class="model-card-header">
                        <div class="model-icon">
                            <i class="fas fa-vector-square"></i>
                        </div>
                        <h3 class="model-name">Support Vector Machine</h3>
                    </div>
                    <div class="model-status" id="status-svm">Initializing...</div>
                    <div class="model-progress">
                        <div class="progress">
                            <div id="progress-svm" class="progress-bar progress-bar-striped progress-bar-animated bg-warning" 
                                 role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Naive Bayes -->
                <div class="model-card in-progress fade-in" id="model-card-naive-bayes">
                    <div class="model-card-header">
                        <div class="model-icon">
                            <i class="fas fa-calculator"></i>
                        </div>
                        <h3 class="model-name">Naive Bayes</h3>
                    </div>
                    <div class="model-status" id="status-naive-bayes">Initializing...</div>
                    <div class="model-progress">
                        <div class="progress">
                            <div id="progress-naive-bayes" class="progress-bar progress-bar-striped progress-bar-animated bg-warning" 
                                 role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <button id="stop-training-btn" class="btn btn-danger stop-training-btn">
                <i class="fas fa-stop-circle mr-2"></i> Stop Training
            </button>
        </div>
    `;
    
    // Add the overlay to the document
    document.body.appendChild(overlay);
    
    // Add event listener to stop training button
    const stopButton = document.getElementById('stop-training-btn');
    if (stopButton) {
        stopButton.addEventListener('click', stopTraining);
    }
    
    // Start polling for training status when training starts
    startPollingTrainingStatus();
}

/**
 * Show the training overlay
 */
function showTrainingOverlay() {
    const overlay = document.getElementById('classification-loading-overlay');
    if (overlay) {
        overlay.classList.add('active');
        // Reset progress bars
        resetProgressBars();
    }
}

/**
 * Reset all progress bars to 0%
 */
function resetProgressBars() {
    // Reset main progress bar
    const mainProgressBar = document.getElementById('main-progress-bar');
    if (mainProgressBar) {
        mainProgressBar.style.width = '0%';
        mainProgressBar.setAttribute('aria-valuenow', 0);
    }
    
    // Reset individual model progress bars
    const modelNames = ['logistic-regression', 'decision-tree', 'random-forest', 'knn', 'svm', 'naive-bayes'];
    modelNames.forEach(name => {
        const progressBar = document.getElementById(`progress-${name}`);
        const statusText = document.getElementById(`status-${name}`);
        
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', 0);
            progressBar.classList.remove('bg-success');
            progressBar.classList.add('bg-warning');
        }
        
        if (statusText) {
            statusText.textContent = 'Initializing...';
        }
        
        const modelCard = document.getElementById(`model-card-${name}`);
        if (modelCard) {
            modelCard.classList.remove('complete');
            modelCard.classList.add('in-progress');
        }
    });
}

/**
 * Update progress for a specific model
 */
function updateModelProgress(modelName, progress, status) {
    const progressBar = document.getElementById(`progress-${modelName}`);
    const statusText = document.getElementById(`status-${modelName}`);
    const modelCard = document.getElementById(`model-card-${modelName}`);
    
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
        
        // Change color when complete
        if (progress >= 100) {
            progressBar.classList.remove('bg-warning', 'progress-bar-animated');
            progressBar.classList.add('bg-success');
            
            if (modelCard) {
                modelCard.classList.remove('in-progress');
                modelCard.classList.add('complete');
            }
        }
    }
    
    if (statusText && status) {
        statusText.textContent = status;
    }
}

/**
 * Update overall progress
 */
function updateOverallProgress(progress, status) {
    const mainProgressBar = document.getElementById('main-progress-bar');
    const mainStatusText = document.getElementById('main-status-text');
    
    if (mainProgressBar) {
        mainProgressBar.style.width = `${progress}%`;
        mainProgressBar.setAttribute('aria-valuenow', progress);
    }
    
    if (mainStatusText && status) {
        mainStatusText.textContent = status;
    }
}

/**
 * Start polling for training status
 */
function startPollingTrainingStatus() {
    // Demo values for models - this would be replaced by real API calls
    const models = [
        { name: 'logistic-regression', steps: [20, 40, 60, 80, 100] },
        { name: 'decision-tree', steps: [25, 50, 75, 100] },
        { name: 'random-forest', steps: [20, 40, 60, 80, 90, 100] },
        { name: 'knn', steps: [33, 66, 100] },
        { name: 'svm', steps: [25, 50, 75, 100] },
        { name: 'naive-bayes', steps: [33, 66, 100] }
    ];
    
    // Overall progress tracking
    let overallCompletion = 0;
    
    // Function to poll API for status
    function pollTrainingStatus() {
        fetch('/api/classification_training_status')
            .then(response => response.json())
            .then(data => {
                // Update model progress based on actual data
                if (data.status === 'in_progress') {
                    // Update model statuses
                    if (data.model_statuses) {
                        Object.keys(data.model_statuses).forEach(modelName => {
                            const modelStatus = data.model_statuses[modelName];
                            updateModelProgress(
                                modelName,
                                modelStatus.progress,
                                modelStatus.status
                            );
                        });
                    }
                    
                    // Update overall progress
                    updateOverallProgress(
                        data.overall_progress,
                        data.overall_status || 'Training in progress...'
                    );
                    
                    // Continue polling
                    setTimeout(pollTrainingStatus, 2000);
                } else if (data.status === 'complete') {
                    // All models complete
                    updateOverallProgress(100, 'Training complete! Redirecting to results...');
                    
                    // Redirect after a short delay
                    setTimeout(() => {
                        window.location.href = data.redirect_url || '/model_selection';
                    }, 1500);
                }
            })
            .catch(error => {
                console.error('Error checking training status:', error);
                
                // Demo mode - simulate training progress for demonstration
                simulateTrainingProgress();
            });
    }
    
    // For demo purposes - simulate progress
    function simulateTrainingProgress() {
        // Simulate progress for each model
        models.forEach(model => {
            const currentStep = model.currentStep || 0;
            if (currentStep < model.steps.length) {
                const progress = model.steps[currentStep];
                let status = 'Training in progress...';
                
                if (progress === 100) {
                    status = 'Model training complete!';
                } else if (progress > 66) {
                    status = 'Evaluating model...';
                } else if (progress > 33) {
                    status = 'Fitting model...';
                }
                
                updateModelProgress(model.name, progress, status);
                model.currentStep = currentStep + 1;
            }
        });
        
        // Calculate overall progress
        const totalSteps = models.reduce((sum, model) => sum + model.steps.length, 0);
        const completedSteps = models.reduce((sum, model) => sum + (model.currentStep || 0), 0);
        overallCompletion = Math.round((completedSteps / totalSteps) * 100);
        
        let overallStatus = 'Training classification models...';
        if (overallCompletion > 90) {
            overallStatus = 'Almost done! Finalizing models...';
        } else if (overallCompletion > 75) {
            overallStatus = 'Models training in progress...';
        } else if (overallCompletion > 50) {
            overallStatus = 'Cross-validating models...';
        } else if (overallCompletion > 25) {
            overallStatus = 'Fitting models to your data...';
        }
        
        updateOverallProgress(overallCompletion, overallStatus);
        
        // Check if all models are done
        const allDone = models.every(model => (model.currentStep || 0) >= model.steps.length);
        
        if (allDone) {
            updateOverallProgress(100, 'Training complete! Redirecting to results...');
            
            // Redirect after a short delay
            setTimeout(() => {
                window.location.href = '/model_selection';
            }, 1500);
        } else {
            // Continue with next step
            setTimeout(simulateTrainingProgress, 2000);
        }
    }
    
    // Start the polling
    pollTrainingStatus();
}

/**
 * Stop the training process
 */
function stopTraining() {
    fetch('/api/stop_classification_training', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'stopped') {
            // Redirect to training page
            window.location.href = '/training';
        } else {
            console.error('Failed to stop training:', data.message);
        }
    })
    .catch(error => {
        console.error('Error stopping training:', error);
        
        // Fallback: redirect anyway
        window.location.href = '/training';
    });
} 