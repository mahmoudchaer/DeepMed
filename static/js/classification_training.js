document.addEventListener('DOMContentLoaded', function() {
    // Initialize the training overlay structure but don't activate it yet
    initTrainingOverlay();
    
    // Handle form submission for the training form
    const trainingForm = document.getElementById('training-form');
    if (trainingForm) {
        trainingForm.addEventListener('submit', function(e) {
            // Show the loading overlay
            showTrainingOverlay();
            
            // Start polling for training status
            startPollingTrainingStatus();
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
}

/**
 * Show the training overlay
 */
function showTrainingOverlay() {
    const overlay = document.getElementById('classification-loading-overlay');
    if (overlay) {
        overlay.classList.add('active');
        
        // Instead of resetting, immediately start showing progress
        const modelNames = ['logistic-regression', 'decision-tree', 'random-forest', 'knn', 'svm', 'naive-bayes'];
        
        // Set initial high progress
        modelNames.forEach(name => {
            const progressBar = document.getElementById(`progress-${name}`);
            const statusText = document.getElementById(`status-${name}`);
            
            if (progressBar) {
                progressBar.style.width = '60%';
                progressBar.setAttribute('aria-valuenow', 60);
            }
            
            if (statusText) {
                statusText.textContent = 'Training in progress...';
            }
        });
        
        // Update overall progress
        const mainProgressBar = document.getElementById('main-progress-bar');
        const mainStatusText = document.getElementById('main-status-text');
        
        if (mainProgressBar) {
            mainProgressBar.style.width = '50%';
            mainProgressBar.setAttribute('aria-valuenow', 50);
        }
        
        if (mainStatusText) {
            mainStatusText.textContent = 'Training models...';
        }
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
    // Immediately show high progress for all models but not 100%
    const modelNames = ['logistic-regression', 'decision-tree', 'random-forest', 'knn', 'svm', 'naive-bayes'];
    
    // Set initial high progress (around 80-95%)
    modelNames.forEach((name, index) => {
        const progressBar = document.getElementById(`progress-${name}`);
        const statusText = document.getElementById(`status-${name}`);
        const modelCard = document.getElementById(`model-card-${name}`);
        
        // Stagger the progress a bit for visual effect
        const progress = 80 + Math.floor(Math.random() * 15);
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
            progressBar.classList.remove('bg-warning', 'progress-bar-animated');
            progressBar.classList.add('bg-success');
        }
        
        if (statusText) {
            statusText.textContent = 'Training in progress...';
        }
        
        if (modelCard) {
            modelCard.classList.remove('in-progress');
            modelCard.classList.add('complete');
        }
    });
    
    // Update overall progress to 85%
    const mainProgressBar = document.getElementById('main-progress-bar');
    const mainStatusText = document.getElementById('main-status-text');
    
    if (mainProgressBar) {
        mainProgressBar.style.width = '85%';
        mainProgressBar.setAttribute('aria-valuenow', 85);
    }
    
    if (mainStatusText) {
        mainStatusText.textContent = 'Models training in progress, finalizing results...';
    }
    
    // Now start polling for the actual status
    pollForActualStatus();
}

/**
 * Poll the server for the actual training status
 */
function pollForActualStatus() {
    fetch('/api/classification_training_status')
        .then(response => response.json())
        .then(data => {
            // Only redirect when the real training is complete
            if (data.status === 'complete') {
                // Update overall progress to 100%
                const mainProgressBar = document.getElementById('main-progress-bar');
                const mainStatusText = document.getElementById('main-status-text');
                
                if (mainProgressBar) {
                    mainProgressBar.style.width = '100%';
                    mainProgressBar.setAttribute('aria-valuenow', 100);
                }
                
                if (mainStatusText) {
                    mainStatusText.textContent = 'Training complete! Redirecting to results...';
                }
                
                // Make all models show 100%
                const modelNames = ['logistic-regression', 'decision-tree', 'random-forest', 'knn', 'svm', 'naive-bayes'];
                modelNames.forEach(name => {
                    const progressBar = document.getElementById(`progress-${name}`);
                    const statusText = document.getElementById(`status-${name}`);
                    
                    if (progressBar) {
                        progressBar.style.width = '100%';
                        progressBar.setAttribute('aria-valuenow', 100);
                    }
                    
                    if (statusText) {
                        statusText.textContent = 'Model training complete!';
                    }
                });
                
                // Redirect after a short delay
                setTimeout(() => {
                    window.location.href = data.redirect_url || '/model_selection';
                }, 800);
            } else {
                // Continue polling until complete
                setTimeout(pollForActualStatus, 1000);
            }
        })
        .catch(error => {
            console.error('Error checking training status:', error);
            // Still poll even on error, as the training might be continuing
            setTimeout(pollForActualStatus, 1000);
        });
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