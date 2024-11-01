document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const errorMessage = document.getElementById('error-message');
    
    // Initialize date input with today's date
    document.getElementById('date').valueAsDate = new Date();
    
    function showLoading() {
        loadingOverlay.style.display = 'flex';
        errorMessage.style.display = 'none';
    }
    
    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }
    
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 5000);
    }
    
    function formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    }
    
    function createModelComparisonChart(predictions) {
        const models = Object.keys(predictions);
        const values = Object.values(predictions);
        
        const data = [{
            type: 'bar',
            x: models,
            y: values,
            marker: {
                color: '#2ecc71'
            }
        }];
        
        const layout = {
            title: 'Model Comparison',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
                color: '#e0e0e0'
            },
            xaxis: {
                title: 'Models',
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            yaxis: {
                title: 'Predicted Sales ($)',
                gridcolor: 'rgba(255,255,255,0.1)'
            }
        };
        
        Plotly.newPlot('model-comparison', data, layout, {responsive: true});
    }
    
    function updateMetricsAnimation(element, value, prefix = '') {
        const duration = 1000;
        const steps = 20;
        const stepValue = value / steps;
        let currentStep = 0;
        
        const interval = setInterval(() => {
            currentStep++;
            const currentValue = stepValue * currentStep;
            element.textContent = prefix + currentValue.toFixed(2);
            
            if (currentStep === steps) {
                clearInterval(interval);
                element.textContent = prefix + value.toFixed(2);
            }
        }, duration / steps);
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
            
            // Update metrics with animation
            const salesElement = document.getElementById('predicted-sales');
            const accuracyElement = document.getElementById('accuracy-score');
            const confidenceElement = document.getElementById('confidence-score');
            
            updateMetricsAnimation(salesElement, data.prediction, '$');
            updateMetricsAnimation(accuracyElement, data.accuracy * 100, '', '%');
            updateMetricsAnimation(confidenceElement, data.confidence * 100, '', '%');
            
            // Update main visualization
            Plotly.newPlot(
                'visualization-container',
                data.plot.data,
                data.plot.layout,
                {responsive: true}
            );
            
            // Create model comparison if available
            if (data.model_comparison) {
                createModelComparisonChart(data.model_comparison);
            }
            
            // Add success feedback
            const btn = document.querySelector('.predict-btn');
            btn.classList.add('btn-success');
            setTimeout(() => {
                btn.classList.remove('btn-success');
            }, 1000);
        })
        .catch(error => {
            hideLoading();
            showError(error.message);
            console.error('Error:', error);
        });
    });
    
    // Add input validation
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.value < 0) {
                this.value = 0;
                showError('Please enter positive values only');
            }
        });
    });
});
