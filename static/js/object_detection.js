document.addEventListener('DOMContentLoaded', function() {
    const yoloForm = document.getElementById('yoloForm');
    const statusBox = document.getElementById('statusBox');
    const statusMessage = document.getElementById('statusMessage');
    const resultMessage = document.getElementById('resultMessage');
    const downloadLink = document.getElementById('downloadLink');
    const submitBtn = document.getElementById('submitBtn');
    let originalFormData = null;
    let trainingId = null;
    
    // Handle form submission
    yoloForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Show processing status
        statusBox.classList.remove('d-none');
        resultMessage.classList.add('d-none');
        statusMessage.textContent = 'Processing... This may take several minutes.';
        submitBtn.disabled = true;
        
        // Get form data and save a copy for potential retry
        originalFormData = new FormData(yoloForm);
        
        // Process the request
        sendTrainingRequest(originalFormData);
    });
    
    // Function to send training request
    function sendTrainingRequest(formData) {
        // Set up a timeout to check if UI needs to be updated even if response is still pending
        const processingTimeout = setTimeout(() => {
            // If we've been processing for over 3 minutes, offer a direct download option
            statusMessage.innerHTML = 'Processing is taking longer than expected. The model may be ready, but your browser is still waiting for the response. <button id="retryDownloadBtn" class="btn btn-sm btn-warning">Try Direct Download</button>';
            
            // Add event listener to the new button
            document.getElementById('retryDownloadBtn').addEventListener('click', function() {
                if (trainingId) {
                    // If we have a training ID, use it to retrieve the model
                    statusMessage.textContent = 'Attempting to retrieve your model...';
                    
                    // Create a download link for the model using the training ID
                    window.location.href = `/api/retrieve_yolo_model/${trainingId}`;
                    
                    // Show completion message
                    setTimeout(() => {
                        statusMessage.textContent = 'Download initiated! If nothing happens, try clicking the download link below.';
                        resultMessage.classList.remove('d-none');
                        submitBtn.disabled = false;
                        
                        // Set the direct download link
                        downloadLink.href = `/api/retrieve_yolo_model/${trainingId}`;
                        downloadLink.download = 'yolov5_model.zip';
                    }, 2000);
                } else {
                    // Without a training ID, we can't retrieve the specific model
                    statusMessage.innerHTML = 'Your training should be complete, but we can\'t automatically retrieve your model. Please try again with your ZIP file.';
                    submitBtn.disabled = false;
                }
            });
        }, 180000); // 3 minutes
        
        // Process the request
        fetch('/api/finetune_yolo', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // Clear the timeout since we got a response
            clearTimeout(processingTimeout);
            
            // Try to get the training ID from the response headers
            trainingId = response.headers.get('X-Training-ID');
            console.log('Training ID:', trainingId);
            
            if (!response.ok) {
                // Handle error response
                return response.json().then(errorData => {
                    throw new Error(errorData.error || 'YOLOv5 fine-tuning failed');
                });
            }
            
            // Process successful response
            return response.blob();
        })
        .then(blob => {
            // Update UI to show completion
            statusMessage.textContent = 'Fine-tuning complete!';
            
            // Create download link for the model
            const url = URL.createObjectURL(blob);
            downloadLink.href = url;
            downloadLink.download = 'yolov5_model.zip';
            
            // Show result message with download link
            resultMessage.classList.remove('d-none');
            
            // Re-enable submit button
            submitBtn.disabled = false;
        })
        .catch(error => {
            // Clear the timeout since we got a response
            clearTimeout(processingTimeout);
            
            // Handle errors
            console.error('Error:', error);
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.classList.remove('text-info');
            statusMessage.classList.add('text-danger');
            
            // If we have a training ID, offer a retrieval option despite the error
            if (trainingId) {
                statusMessage.innerHTML += `<br>Your model may still be available. <button id="retrieveBtn" class="btn btn-sm btn-primary mt-2">Try Retrieving Model</button>`;
                document.getElementById('retrieveBtn').addEventListener('click', function() {
                    window.location.href = `/api/retrieve_yolo_model/${trainingId}`;
                });
            } else {
                // Add a retry button
                statusMessage.innerHTML += '<br><button id="retryBtn" class="btn btn-sm btn-primary mt-2">Retry</button>';
                document.getElementById('retryBtn').addEventListener('click', function() {
                    // Reload the page to start fresh
                    window.location.reload();
                });
            }
            
            // Re-enable submit button
            submitBtn.disabled = false;
        });
    }
}); 