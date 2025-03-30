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
            
            // Log all headers for debugging
            console.log('Response headers:');
            response.headers.forEach((value, name) => {
                console.log(`${name}: ${value}`);
                // Check for metrics in any header that might contain them
                if (name.toLowerCase().includes('metrics') || name.toLowerCase().includes('training')) {
                    try {
                        const parsedValue = JSON.parse(value);
                        if (typeof parsedValue === 'object') {
                            metrics = parsedValue;
                        }
                    } catch (error) {
                        console.error(`Error parsing ${name} header:`, error);
                    }
                }
            });
            
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
                
                // Check for different possible metric names
                if ('training_accuracy' in metrics) {
                    metricsHtml += `<li class="list-group-item">Training Accuracy: ${metrics.training_accuracy.toFixed(2)}%</li>`;
                } else if ('train_accuracy' in metrics) {
                    metricsHtml += `<li class="list-group-item">Training Accuracy: ${metrics.train_accuracy.toFixed(2)}%</li>`;
                }
                
                if ('validation_accuracy' in metrics) {
                    metricsHtml += `<li class="list-group-item">Validation Accuracy: ${metrics.validation_accuracy.toFixed(2)}%</li>`;
                } else if ('val_accuracy' in metrics) {
                    metricsHtml += `<li class="list-group-item">Validation Accuracy: ${metrics.val_accuracy.toFixed(2)}%</li>`;
                }
                
                if ('test_accuracy' in metrics) {
                    metricsHtml += `<li class="list-group-item">Test Accuracy: ${metrics.test_accuracy.toFixed(2)}%</li>`;
                }
                
                if ('final_train_loss' in metrics) {
                    metricsHtml += `<li class="list-group-item">Final Training Loss: ${metrics.final_train_loss.toFixed(4)}</li>`;
                } else if ('final_loss' in metrics) {
                    metricsHtml += `<li class="list-group-item">Final Loss: ${metrics.final_loss.toFixed(4)}</li>`;
                }
                
                if ('final_val_loss' in metrics) {
                    metricsHtml += `<li class="list-group-item">Validation Loss: ${metrics.final_val_loss.toFixed(4)}</li>`;
                }
                
                if ('test_loss' in metrics) {
                    metricsHtml += `<li class="list-group-item">Test Loss: ${metrics.test_loss.toFixed(4)}</li>`;
                }
                
                metricsHtml += '</ul></div>';
                
                // Parameters in second column
                metricsHtml += '<div class="col-md-6"><h6>Training Parameters:</h6><ul class="list-group">';
                if ('num_classes' in metrics) {
                    metricsHtml += `<li class="list-group-item">Classes: ${metrics.num_classes}</li>`;
                }
                
                if ('num_total_images' in metrics) {
                    metricsHtml += `<li class="list-group-item">Total Images: ${metrics.num_total_images}</li>`;
                } else if ('num_images' in metrics) {
                    metricsHtml += `<li class="list-group-item">Images Processed: ${metrics.num_images}</li>`;
                }
                
                if ('num_train_images' in metrics) {
                    metricsHtml += `<li class="list-group-item">Training Images: ${metrics.num_train_images}</li>`;
                }
                
                if ('num_val_images' in metrics) {
                    metricsHtml += `<li class="list-group-item">Validation Images: ${metrics.num_val_images}</li>`;
                }
                
                if ('num_test_images' in metrics) {
                    metricsHtml += `<li class="list-group-item">Test Images: ${metrics.num_test_images}</li>`;
                }
                
                if ('epochs' in metrics) {
                    metricsHtml += `<li class="list-group-item">Epochs: ${metrics.epochs}</li>`;
                }
                
                if ('batch_size' in metrics) {
                    metricsHtml += `<li class="list-group-item">Batch Size: ${metrics.batch_size}</li>`;
                }
                
                if ('device' in metrics) {
                    metricsHtml += `<li class="list-group-item">Device: ${metrics.device}</li>`;
                }
                
                if ('learning_rate' in metrics) {
                    metricsHtml += `<li class="list-group-item">Learning Rate: ${metrics.learning_rate}</li>`;
                }
                
                if ('training_level' in metrics) {
                    metricsHtml += `<li class="list-group-item">Training Level: ${metrics.training_level}</li>`;
                }
                
                metricsHtml += '</ul></div>';
                
                metricsHtml += '</div>'; // End row
            } else {
                console.warn('No metrics available or metrics object is empty', metrics);
                metricsHtml += '<p>No metrics available. This could happen if the model training service did not return complete metrics information.</p>';
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
