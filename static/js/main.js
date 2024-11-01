document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const errorMessage = document.getElementById('error-message');
    
    if (!form) {
        console.error('Form element not found');
        return;
    }
    
    const dateInput = document.getElementById('date');
    if (dateInput) {
        dateInput.valueAsDate = new Date();
    }
    
    function showLoading() {
        if (loadingOverlay) {
            loadingOverlay.style.display = 'flex';
        }
        if (errorMessage) {
            errorMessage.style.display = 'none';
        }
    }
    
    function hideLoading() {
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }
    
    function showError(message) {
        if (errorMessage) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
            setTimeout(() => {
                errorMessage.style.display = 'none';
            }, 5000);
        }
    }
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        showLoading();
        
        const formData = new FormData(this);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Prediction failed');
                });
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
            
            document.getElementById('predicted-sales').textContent = 
                `$${parseFloat(data.prediction).toFixed(2)}`;
            document.getElementById('accuracy-score').textContent = 
                `${(data.accuracy * 100).toFixed(1)}%`;
            document.getElementById('confidence-score').textContent = 
                `${(data.confidence * 100).toFixed(1)}%`;
            
            if (data.plot) {
                Plotly.newPlot(
                    'visualization-container',
                    data.plot.data,
                    data.plot.layout,
                    {responsive: true}
                );
            }
        })
        .catch(error => {
            hideLoading();
            showError(error.message);
            console.error('Error:', error);
        });
    });
});
