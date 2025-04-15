document.addEventListener('DOMContentLoaded', function() {
    const anomalyForm = document.getElementById('anomalyForm');
    const statusBox = document.getElementById('statusBox');
    const statusMessage = document.getElementById('statusMessage');
    const resultMessage = document.getElementById('resultMessage');
    const downloadLink = document.getElementById('downloadLink');
    const submitBtn = document.getElementById('submitBtn');
    
    // Store form data globally for retry functionality
    let lastFormData = null;
    
    // Handle form submission
    anomalyForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Show processing status
        statusBox.classList.remove('d-none');
        resultMessage.classList.add('d-none');
        statusMessage.textContent = 'Processing... This may take several minutes.';
        statusMessage.classList.remove('text-danger');
        statusMessage.classList.add('text-info');
        submitBtn.disabled = true;
        
        // Add dots animation
        let dots = 0;
        const dotsInterval = setInterval(() => {
            dots = (dots + 1) % 4;
            statusMessage.textContent = 'Processing' + '.'.repeat(dots) + ' This may take several minutes.';
        }, 500);
        
        // Get form data and store it for potential retries
        const formData = new FormData(anomalyForm);
        lastFormData = formData;
        
        // Process the request
        submitTrainingJob(formData, dotsInterval);
    });
    
    // Function to submit the training job
    function submitTrainingJob(formData, dotsInterval) {
        fetch('/api/train_anomaly', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            clearInterval(dotsInterval);
            
            if (!response.ok) {
                // Handle error response
                return response.json().then(errorData => {
                    throw new Error(errorData.error || 'Anomaly detection training failed');
                });
            }
            
            // Process successful response
            return response.blob();
        })
        .then(blob => {
            // Update UI to show completion
            statusMessage.textContent = 'Training complete!';
            
            // Create download link for the model
            const url = URL.createObjectURL(blob);
            downloadLink.href = url;
            downloadLink.download = 'anomaly_detection_model.zip';
            
            // Show result message with download link
            resultMessage.classList.remove('d-none');
            
            // Re-enable submit button
            submitBtn.disabled = false;
            
            // Trigger download automatically
            downloadLink.click();
        })
        .catch(error => {
            clearInterval(dotsInterval);
            
            // Handle errors
            console.error('Error:', error);
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.classList.remove('text-info');
            statusMessage.classList.add('text-danger');
            
            // Add retry option
            const retryDiv = document.createElement('div');
            retryDiv.className = 'mt-3';
            retryDiv.innerHTML = `
                <p>If your training was in progress but timed out, you can try to:</p>
                <button id="retryBtn" class="btn btn-primary btn-sm me-2">Try Again</button>
                <button id="checkStatusBtn" class="btn btn-outline-secondary btn-sm">Check Service Status</button>
            `;
            
            // Add retry div if not already there
            if (!document.getElementById('retryBtn')) {
                statusMessage.parentNode.appendChild(retryDiv);
                
                // Retry button handler
                document.getElementById('retryBtn').addEventListener('click', function() {
                    // Remove error messages
                    statusMessage.textContent = 'Processing... Retrying submission.';
                    statusMessage.classList.remove('text-danger');
                    statusMessage.classList.add('text-info');
                    
                    // Start a new dots animation
                    let dots = 0;
                    const newDotsInterval = setInterval(() => {
                        dots = (dots + 1) % 4;
                        statusMessage.textContent = 'Processing' + '.'.repeat(dots) + ' This may take several minutes.';
                    }, 500);
                    
                    // Remove retry div
                    retryDiv.remove();
                    
                    // Try again
                    if (lastFormData) {
                        submitTrainingJob(lastFormData, newDotsInterval);
                    } else {
                        clearInterval(newDotsInterval);
                        statusMessage.textContent = 'Error: Could not retry. Please refresh the page and try again.';
                        statusMessage.classList.remove('text-info');
                        statusMessage.classList.add('text-danger');
                        submitBtn.disabled = false;
                    }
                });
                
                // Check status button handler
                document.getElementById('checkStatusBtn').addEventListener('click', function() {
                    fetch('/health')
                        .then(response => response.json())
                        .then(data => {
                            alert(`Service Status: ${data.status || 'Running'}\nAll systems are operational.`);
                        })
                        .catch(error => {
                            alert('Could not check service status. The server might be overloaded.');
                        });
                });
            }
            
            // Re-enable submit button
            submitBtn.disabled = false;
        });
    }
}); 