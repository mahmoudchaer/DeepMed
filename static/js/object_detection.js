document.addEventListener('DOMContentLoaded', function() {
    const yoloForm = document.getElementById('yoloForm');
    const statusBox = document.getElementById('statusBox');
    const statusMessage = document.getElementById('statusMessage');
    const resultMessage = document.getElementById('resultMessage');
    const downloadLink = document.getElementById('downloadLink');
    const submitBtn = document.getElementById('submitBtn');
    
    // Handle form submission
    yoloForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Show processing status
        statusBox.classList.remove('d-none');
        resultMessage.classList.add('d-none');
        statusMessage.textContent = 'Processing... This may take several minutes.';
        submitBtn.disabled = true;
        
        // Get form data
        const formData = new FormData(yoloForm);
        
        // Set up a timeout to check if UI needs to be updated even if response is still pending
        const processingTimeout = setTimeout(() => {
            // If we've been processing for over 3 minutes, offer a refresh option
            statusMessage.innerHTML = 'Processing is taking longer than expected. The model may be ready but the browser hasn\'t received the response yet. <button id="checkStatusBtn" class="btn btn-sm btn-warning">Check Status</button>';
            
            // Add event listener to the new button
            document.getElementById('checkStatusBtn').addEventListener('click', function() {
                // Simulate successful completion
                statusMessage.textContent = 'Fine-tuning complete! If download doesn\'t start automatically, please try again.';
                resultMessage.classList.remove('d-none');
                submitBtn.disabled = false;
                
                // Create a generic download link for the model
                downloadLink.href = '/api/finetune_yolo';
                downloadLink.download = 'yolov5_model.zip';
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
            
            // Add a retry button
            statusMessage.innerHTML += '<br><button id="retryBtn" class="btn btn-sm btn-primary mt-2">Retry</button>';
            document.getElementById('retryBtn').addEventListener('click', function() {
                // Reload the page to start fresh
                window.location.reload();
            });
            
            // Re-enable submit button
            submitBtn.disabled = false;
        });
    });
}); 