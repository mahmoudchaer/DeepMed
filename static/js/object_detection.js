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
        
        // Process the request
        fetch('/api/finetune_yolo', {
            method: 'POST',
            body: formData
        })
        .then(response => {
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