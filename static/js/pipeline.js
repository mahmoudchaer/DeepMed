document.addEventListener('DOMContentLoaded', function() {
    const pipelineForm = document.getElementById('pipelineForm');
    const statusBox = document.getElementById('statusBox');
    const statusMessage = document.getElementById('statusMessage');
    const resultMessage = document.getElementById('resultMessage');
    const downloadLink = document.getElementById('downloadLink');
    const submitBtn = document.getElementById('submitBtn');
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
        statusMessage.textContent = 'Processing... This may take several minutes.';
        submitBtn.disabled = true;
        
        // Get form data
        const formData = new FormData(pipelineForm);
        
        // Convert checkbox value to string 'true'/'false' for backend
        if (performAugmentation.checked) {
            formData.set('performAugmentation', 'true');
        } else {
            formData.set('performAugmentation', 'false');
            // Remove augmentation parameters if not performing augmentation
            formData.delete('augmentationLevel');
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
            statusMessage.textContent = 'Processing complete!';
            
            // Create download link for the model
            const url = URL.createObjectURL(blob);
            downloadLink.href = url;
            downloadLink.download = 'model_package.zip';
            
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
            
            // Re-enable submit button
            submitBtn.disabled = false;
        });
    });
    
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
}); 