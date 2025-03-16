document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('train-model-form');
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const statusDiv = document.getElementById('training-status');
        
        // Get training level to customize the message
        const trainingLevel = document.getElementById('trainingLevel').value;
        const numClasses = document.getElementById('numClasses').value;
        
        // More detailed status message based on training level
        let statusMessage = '';
        if (trainingLevel >= 4) {
            statusMessage = '<div class="alert alert-info">Training in progress (Level ' + trainingLevel + 
                            ')... This may take several minutes. Please be patient.</div>';
        } else {
            statusMessage = '<div class="alert alert-info">Training in progress (Level ' + trainingLevel + 
                            ')... This may take a moment.</div>';
        }
        
        statusDiv.innerHTML = statusMessage;
        
        const formData = new FormData(form);
        fetch('/api/train_model', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // Extract metrics from headers
            let metrics = {};
            if (response.headers.has('X-Training-Metrics')) {
                try {
                    metrics = JSON.parse(response.headers.get('X-Training-Metrics'));
                } catch (error) {
                    console.error('Error parsing metrics:', error);
                }
            }
            
            if (!response.ok) {
                return response.json().then(errData => { throw new Error(errData.error || 'Training failed'); });
            }
            
            // Store metrics for later use
            window.trainingMetrics = metrics;
            return response.blob().then(blob => ({ blob, metrics }));
        })
        .then(({ blob, metrics }) => {
            // Create a download link for the trained model file
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'trained_model.pt';
            a.className = 'btn btn-success mb-3';
            a.textContent = 'Download Trained Model';
            
            // Create a metrics display
            const metricsDisplay = document.createElement('div');
            metricsDisplay.className = 'card mb-3';
            
            let metricsHtml = '<div class="card-header"><h5>Training Metrics</h5></div><div class="card-body">';
            
            if (metrics && Object.keys(metrics).length > 0) {
                // Format metrics information
                metricsHtml += '<div class="row">';
                
                // Main metrics in first column
                metricsHtml += '<div class="col-md-6"><h6>Training Results:</h6><ul class="list-group">';
                if ('training_accuracy' in metrics) {
                    metricsHtml += `<li class="list-group-item">Accuracy: ${metrics.training_accuracy.toFixed(2)}%</li>`;
                }
                if ('final_loss' in metrics) {
                    metricsHtml += `<li class="list-group-item">Final Loss: ${metrics.final_loss.toFixed(4)}</li>`;
                }
                metricsHtml += '</ul></div>';
                
                // Parameters in second column
                metricsHtml += '<div class="col-md-6"><h6>Training Parameters:</h6><ul class="list-group">';
                if ('num_classes' in metrics) {
                    metricsHtml += `<li class="list-group-item">Classes: ${metrics.num_classes}</li>`;
                }
                if ('num_images' in metrics) {
                    metricsHtml += `<li class="list-group-item">Images Processed: ${metrics.num_images}</li>`;
                }
                if ('epochs' in metrics) {
                    metricsHtml += `<li class="list-group-item">Epochs: ${metrics.epochs}</li>`;
                }
                if ('device' in metrics) {
                    metricsHtml += `<li class="list-group-item">Device: ${metrics.device}</li>`;
                }
                metricsHtml += '</ul></div>';
                
                metricsHtml += '</div>'; // End row
            } else {
                metricsHtml += '<p>No metrics available</p>';
            }
            
            metricsHtml += '</div>'; // End card-body
            metricsDisplay.innerHTML = metricsHtml;
            
            // Update the status div with success message and components
            statusDiv.innerHTML = '<div class="alert alert-success">Training complete!</div>';
            statusDiv.appendChild(metricsDisplay);
            statusDiv.appendChild(a);
        })
        .catch(error => {
            statusDiv.innerHTML = `<div class="alert alert-danger">Error during training: ${error.message}</div>`;
        });
    });
});
