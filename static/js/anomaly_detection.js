document.addEventListener('DOMContentLoaded', function() {
    const anomalyForm = document.getElementById('anomalyForm');
    const statusBox = document.getElementById('statusBox');
    const statusMessage = document.getElementById('statusMessage');
    const resultMessage = document.getElementById('resultMessage');
    const downloadLink = document.getElementById('downloadLink');
    const submitBtn = document.getElementById('submitBtn');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const timeEstimate = document.getElementById('timeEstimate');
    const timeRemaining = document.getElementById('timeRemaining');
    
    // Global variable to track request status
    let requestInProgress = false;
    let requestStartTime = null;
    let requestTimeout = null;
    
    // Training time estimates based on level selection
    const levelTimeEstimates = {
        '1': 5,    // Level 1: ~5 minutes
        '2': 10,   // Level 2: ~10 minutes
        '3': 20,   // Level 3: ~20 minutes
        '4': 35,   // Level 4: ~35 minutes
        '5': 60    // Level 5: ~60 minutes
    };
    
    // Handle form submission
    anomalyForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Prevent multiple submissions
        if (requestInProgress) {
            return;
        }
        
        // Get form data
        const formData = new FormData(anomalyForm);
        const level = formData.get('level');
        
        // Show processing status
        statusBox.classList.remove('d-none');
        resultMessage.classList.add('d-none');
        statusMessage.textContent = 'Processing... This may take several minutes.';
        statusMessage.classList.remove('text-danger');
        statusMessage.classList.add('text-info');
        submitBtn.disabled = true;
        
        // Show progress indicators
        progressContainer.classList.remove('d-none');
        timeEstimate.classList.remove('d-none');
        
        // Estimate completion time based on training level
        const estimatedMinutes = levelTimeEstimates[level] || 20;
        timeRemaining.textContent = `approximately ${estimatedMinutes} minutes`;
        
        requestInProgress = true;
        requestStartTime = Date.now();
        
        // Create a progress indicator
        updateProgressIndicator(estimatedMinutes);
        
        // Create direct download link
        const directLinkMessage = document.createElement('div');
        directLinkMessage.className = 'mt-3 direct-download-message';
        directLinkMessage.innerHTML = `
            <p>If the download doesn't start automatically after training completes, you can use this direct link:</p>
            <button id="directDownloadBtn" class="btn btn-outline-primary btn-sm mt-2">Download Model Directly</button>
        `;
        
        // Set up timeout to show direct download option after 3 minutes
        setTimeout(() => {
            if (requestInProgress) {
                statusBox.appendChild(directLinkMessage);
                document.getElementById('directDownloadBtn').addEventListener('click', function() {
                    // Get form values for direct API call
                    const zipFile = formData.get('zipFile').name;
                    const level = formData.get('level');
                    const imageSize = formData.get('image_size');
                    
                    // Create a link to the API endpoint
                    const a = document.createElement('a');
                    a.href = `/api/train_anomaly?filename=${encodeURIComponent(zipFile)}&level=${level}&image_size=${imageSize}`;
                    a.download = 'anomaly_detection_model.zip';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                });
            }
        }, 180000); // 3 minutes
        
        // Process the request with a longer timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 1800000); // 30 minute timeout
        
        fetch('/api/train_anomaly', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeoutId);
            
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
            requestInProgress = false;
            clearInterval(requestTimeout);
            statusMessage.textContent = 'Training complete!';
            
            // Hide progress indicators
            progressContainer.classList.add('d-none');
            timeEstimate.classList.add('d-none');
            
            // Remove direct download message if it exists
            const directDownloadMsg = document.querySelector('.direct-download-message');
            if (directDownloadMsg) {
                directDownloadMsg.remove();
            }
            
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
            // Handle errors
            requestInProgress = false;
            clearInterval(requestTimeout);
            console.error('Error:', error);
            
            // Hide progress indicators
            progressContainer.classList.add('d-none');
            timeEstimate.classList.add('d-none');
            
            let errorMessage = error.message;
            if (error.name === 'AbortError') {
                errorMessage = 'Request timed out. The server might still be processing your request. You can try the direct download button below.';
                
                // Add direct download button if not already added
                if (!document.querySelector('.direct-download-message')) {
                    statusBox.appendChild(directLinkMessage);
                    document.getElementById('directDownloadBtn').addEventListener('click', function() {
                        // Get form values for direct API call
                        const zipFile = formData.get('zipFile').name;
                        const level = formData.get('level');
                        const imageSize = formData.get('image_size');
                        
                        // Create a link to the API endpoint
                        const a = document.createElement('a');
                        a.href = `/api/train_anomaly?filename=${encodeURIComponent(zipFile)}&level=${level}&image_size=${imageSize}`;
                        a.download = 'anomaly_detection_model.zip';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    });
                }
            }
            
            statusMessage.textContent = `Error: ${errorMessage}`;
            statusMessage.classList.remove('text-info');
            statusMessage.classList.add('text-danger');
            
            // Re-enable submit button
            submitBtn.disabled = false;
        });
    });
    
    // Function to update progress indicator
    function updateProgressIndicator(estimatedMinutes) {
        let dots = '';
        let seconds = 0;
        const totalEstimatedSeconds = estimatedMinutes * 60;
        
        requestTimeout = setInterval(() => {
            seconds = Math.floor((Date.now() - requestStartTime) / 1000);
            dots = (dots.length >= 3) ? '' : dots + '.';
            
            // Calculate minutes and seconds
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            const timeString = `${minutes}m ${remainingSeconds}s`;
            
            // Update progress bar (capped at 99% until complete)
            const progressPercentage = Math.min(99, (seconds / totalEstimatedSeconds) * 100);
            progressBar.style.width = `${progressPercentage}%`;
            
            // Update remaining time estimate
            const remainingTimeSeconds = Math.max(0, totalEstimatedSeconds - seconds);
            const remainingMinutes = Math.floor(remainingTimeSeconds / 60);
            const remainingSecs = remainingTimeSeconds % 60;
            
            if (remainingTimeSeconds > 0) {
                if (remainingMinutes > 0) {
                    timeRemaining.textContent = `approximately ${remainingMinutes}m ${remainingSecs}s`;
                } else {
                    timeRemaining.textContent = `less than a minute`;
                }
            } else {
                timeRemaining.textContent = `finalizing model...`;
            }
            
            // Update status message with dots animation
            statusMessage.textContent = `Processing${dots} (${timeString}) This may take several minutes.`;
            
            // After 5 minutes, add a note about large files
            if (seconds > 300 && requestInProgress) {
                statusMessage.textContent = `Processing${dots} (${timeString}) Large datasets can take up to 30 minutes. Please be patient.`;
            }
        }, 1000);
    }
}); 