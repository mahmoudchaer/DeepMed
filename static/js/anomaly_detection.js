document.addEventListener('DOMContentLoaded', function() {
    const anomalyForm = document.getElementById('anomalyForm');
    const statusBox = document.getElementById('statusBox');
    const statusMessage = document.getElementById('statusMessage');
    const resultMessage = document.getElementById('resultMessage');
    const downloadLink = document.getElementById('downloadLink');
    const submitBtn = document.getElementById('submitBtn');
    
    // Handle form submission
    anomalyForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Show processing status
        statusBox.classList.remove('d-none');
        resultMessage.classList.add('d-none');
        statusMessage.textContent = 'Processing... This may take several minutes.';
        submitBtn.disabled = true;
        
        // Get form data
        const formData = new FormData(anomalyForm);
        
        // Process the request
        fetch('/api/train_anomaly', {
            method: 'POST',
            body: formData
        })
        .then(response => {
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
        })
        .catch(error => {
            // Handle errors
            console.error('Error:', error);
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.classList.remove('text-info');
            statusMessage.classList.add('text-danger');
            
            // Re-enable submit button
            submitBtn.disabled = false;
        });
    });
}); 