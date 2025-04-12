document.addEventListener('DOMContentLoaded', function() {
    const pipelineForm = document.getElementById('pipelineForm');
    const statusBox = document.getElementById('statusBox');
    const statusMessage = document.getElementById('statusMessage');
    const pipelineProgress = document.getElementById('pipelineProgress');
    const resultMessage = document.getElementById('resultMessage');
    const downloadLink = document.getElementById('downloadLink');
    const submitBtn = document.getElementById('submitBtn');
    const stepIndicator = document.getElementById('stepIndicator');
    const performAugmentation = document.getElementById('performAugmentation');
    const augmentationOptions = document.getElementById('augmentationOptions');
    const metricsContainer = document.getElementById('metricsContainer');
    const metricsTable = document.getElementById('metricsTable');
    
    // Toggle augmentation options visibility based on checkbox
    performAugmentation.addEventListener('change', function() {
        if (this.checked) {
            augmentationOptions.style.display = 'block';
        } else {
            augmentationOptions.style.display = 'none';
        }
    });
    
    // Handle form submission
    pipelineForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Show processing status
        statusBox.classList.remove('d-none');
        resultMessage.classList.add('d-none');
        metricsContainer.classList.add('d-none');
        statusMessage.textContent = 'Starting pipeline process...';
        pipelineProgress.style.width = '10%';
        submitBtn.disabled = true;
        
        // Update step indicator to show we're at upload stage
        updateStepIndicator(1);
        
        // Get form data
        const formData = new FormData(pipelineForm);
        
        // Convert checkbox value to string 'true'/'false' for backend
        if (performAugmentation.checked) {
            formData.set('performAugmentation', 'true');
        } else {
            formData.set('performAugmentation', 'false');
            // Remove augmentation parameters if not performing augmentation
            formData.delete('augmentationLevel');
            formData.delete('numAugmentations');
        }
        
        // Process the pipeline
        fetch('/api/pipeline', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                // Handle error response
                return response.json().then(errorData => {
                    throw new Error(errorData.error || 'Pipeline processing failed');
                });
            }
            
            // Get training metrics from header if available
            const metricsHeader = response.headers.get('X-Training-Metrics');
            let metrics = null;
            if (metricsHeader) {
                try {
                    metrics = JSON.parse(metricsHeader);
                } catch (e) {
                    console.error('Error parsing metrics:', e);
                }
            }
            
            // Process successful response
            return response.blob().then(blob => {
                return { blob, metrics };
            });
        })
        .then(({ blob, metrics }) => {
            // Update UI to show completion
            pipelineProgress.style.width = '100%';
            statusMessage.textContent = 'Pipeline completed successfully!';
            updateStepIndicator(4); // Set to complete step
            
            // Create download link for the model
            const url = URL.createObjectURL(blob);
            downloadLink.href = url;
            downloadLink.download = 'trained_model.pt';
            
            // Show result message with download link
            resultMessage.classList.remove('d-none');
            
            // Display metrics if available
            if (metrics) {
                displayMetrics(metrics);
            }
            
            // Re-enable submit button
            submitBtn.disabled = false;
        })
        .catch(error => {
            // Handle errors
            console.error('Error:', error);
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.classList.remove('text-info');
            statusMessage.classList.add('text-danger');
            pipelineProgress.classList.remove('bg-info', 'progress-bar-animated', 'progress-bar-striped');
            pipelineProgress.classList.add('bg-danger');
            pipelineProgress.style.width = '100%';
            
            // Re-enable submit button
            submitBtn.disabled = false;
        });
        
        // Simulate progress updates for better UX
        simulateProgress(performAugmentation.checked);
    });
    
    // Function to update step indicator
    function updateStepIndicator(step) {
        // Reset all steps to secondary
        const steps = stepIndicator.querySelectorAll('.badge');
        steps.forEach((badge, index) => {
            if (index + 1 <= step) {
                badge.classList.remove('bg-secondary');
                badge.classList.add('bg-primary');
            } else {
                badge.classList.remove('bg-primary');
                badge.classList.add('bg-secondary');
            }
        });
    }
    
    // Function to display training metrics
    function displayMetrics(metrics) {
        if (!metrics) return;
        
        metricsContainer.classList.remove('d-none');
        metricsTable.innerHTML = ''; // Clear existing content
        
        // Add each metric to the table
        Object.entries(metrics).forEach(([key, value]) => {
            // Format the key for display
            const formattedKey = key
                .replace(/_/g, ' ')
                .split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
            
            // Format the value (round numbers to 2 decimal places)
            let formattedValue = value;
            if (typeof value === 'number') {
                formattedValue = Number.isInteger(value) ? value : value.toFixed(2);
            }
            
            // Add row to table
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${formattedKey}</strong></td>
                <td>${formattedValue}</td>
            `;
            metricsTable.appendChild(row);
        });
    }
    
    // Function to simulate progress updates for better UX
    function simulateProgress(includeAugmentation) {
        let progress = 10; // Start at 10%
        const totalSteps = includeAugmentation ? 30 : 20; // More steps if augmentation is included
        const interval = setInterval(() => {
            // Increment progress
            progress += 1;
            
            // Update progress bar
            pipelineProgress.style.width = `${progress}%`;
            
            // Update step indicator based on progress
            if (progress >= 15 && progress < 50 && includeAugmentation) {
                statusMessage.textContent = 'Augmenting dataset...';
                updateStepIndicator(2);
            } else if ((progress >= 50 && includeAugmentation) || 
                       (progress >= 15 && !includeAugmentation)) {
                statusMessage.textContent = 'Training model...';
                updateStepIndicator(3);
            }
            
            // Stop when we reach 95% (real completion will set to 100%)
            if (progress >= 95) {
                clearInterval(interval);
            }
        }, totalSteps * 100); // Speed depends on whether augmentation is included
        
        // Store the interval ID so we can clear it if needed
        window.progressInterval = interval;
    }
}); 