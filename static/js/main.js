// Model descriptions
const modelDescriptions = {
    'random_forest': '* Ensemble learning method using multiple decision trees. Best for balanced performance and handling non-linear patterns.',
    'gradient_boost': '* Sequential ensemble method that builds trees to correct previous errors. High accuracy with proper tuning.',
    'xgboost': '* Advanced implementation of gradient boosting with better regularization and parallel processing.',
    'lightgbm': '* Gradient boosting framework using leaf-wise tree growth. Fast training and memory efficient.',
    'svr': '* Kernel-based method that maps data to higher dimensions. Effective for non-linear relationships.',
    'neural_network': '* Deep learning model with multiple layers. Automatically learns complex patterns from data.'
};

document.addEventListener('DOMContentLoaded', function() {
    console.log('Application initialized');
    
    // Initialize Socket.IO if available
    if (typeof io !== 'undefined') {
        const socket = io();
        socket.on('connect', () => {
            console.log('WebSocket connected');
        });
        socket.on('data_update', (data) => {
            console.log('Received data update:', data);
        });
    }
    
    // Initialize form handling
    const form = document.querySelector('#prediction-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }

    // Initialize model description handling
    const mlModelSelect = document.getElementById('ml_model');
    if (mlModelSelect) {
        mlModelSelect.addEventListener('change', updateModelDescription);
        updateModelDescription();
    }
});

async function handleFormSubmit(e) {
    e.preventDefault();
    console.log('Form submitted');
    
    if (!validateForm()) {
        return;
    }

    // Show loading overlay
    document.querySelector('.loading-overlay').style.display = 'block';
    
    try {
        const formData = new FormData(e.target);
        const jsonData = {};
        formData.forEach((value, key) => {
            jsonData[key] = value;
        });
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(jsonData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Prediction result:', data);
        
        updateDashboard(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    } finally {
        document.querySelector('.loading-overlay').style.display = 'none';
    }
}

function updateDashboard(data) {
    // Update prediction display
    const predictedSales = document.getElementById('predicted-sales');
    if (predictedSales) {
        predictedSales.textContent = `$${parseFloat(data.prediction).toFixed(2)}`;
    }
    
    // Update visualization if available
    if (data.plot) {
        Plotly.newPlot('visualization-container', data.plot.data, data.plot.layout);
    }
}

function showError(message) {
    showMessage(message, 'danger');
}

function showMessage(message, type = 'info') {
    const messageDiv = document.getElementById('message-container') || createMessageContainer();
    messageDiv.textContent = message;
    messageDiv.className = `alert alert-${type} mt-3`;
    setTimeout(() => messageDiv.remove(), 5000);
}

function createMessageContainer() {
    const container = document.createElement('div');
    container.id = 'message-container';
    document.querySelector('form').appendChild(container);
    return container;
}

function updateModelDescription() {
    const selectedModel = document.getElementById('ml_model').value;
    const descriptionElement = document.getElementById('model-description');
    if (descriptionElement) {
        descriptionElement.textContent = modelDescriptions[selectedModel] || '';
    }
}

function validateForm() {
    const mlModel = document.getElementById('ml_model')?.value;
    const traffic = document.getElementById('traffic')?.value;
    const marketing = document.getElementById('marketing')?.value;
    const advertising = document.getElementById('advertising')?.value;
    const social = document.getElementById('social')?.value;

    if (!mlModel) {
        showMessage('Please select a model', 'warning');
        return false;
    }

    if (!traffic || !marketing || !advertising || !social) {
        showMessage('Please fill in all required fields', 'warning');
        return false;
    }

    return true;
}
