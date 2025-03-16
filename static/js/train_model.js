document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('train-model-form');
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const statusDiv = document.getElementById('training-status');
        statusDiv.innerHTML = '<div class="alert alert-info">Training in progress... This may take a moment.</div>';
        const formData = new FormData(form);
        fetch('/api/train_model', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => { throw new Error(errData.error || 'Training failed'); });
            }
            return response.blob();
        })
        .then(blob => {
            // Create a download link for the trained model file.
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'trained_model.pt';
            a.className = 'btn btn-success';
            a.textContent = 'Download Trained Model';
            statusDiv.innerHTML = '';
            statusDiv.appendChild(a);
        })
        .catch(error => {
            statusDiv.innerHTML = `<div class="alert alert-danger">Error during training: ${error.message}</div>`;
        });
    });
});
