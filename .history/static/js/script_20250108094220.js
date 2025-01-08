const modelDescriptions = {
    'random_forest': '* Ensemble learning method using multiple decision trees. Best for balanced performance and handling non-linear patterns.',
    'gradient_boost': '* Sequential ensemble method that builds trees to correct previous errors. High accuracy with proper tuning.',
    'xgboost': '* Advanced implementation of gradient boosting with better regularization and parallel processing.',
    'lightgbm': '* Gradient boosting framework using leaf-wise tree growth. Fast training and memory efficient.',
    'svr': '* Kernel-based method that maps data to higher dimensions. Effective for non-linear relationships.',
    'neural_network': '* Deep learning model with multiple layers. Automatically learns complex patterns from data.'
};

function updateModelDescription() {
    const selectedModel = document.getElementById('ml_model').value;
    const descriptionElement = document.getElementById('model-description');
    descriptionElement.textContent = modelDescriptions[selectedModel] || '';
}

document.addEventListener('DOMContentLoaded', function() {
    const mlModelSelect = document.getElementById('ml_model');
    if (mlModelSelect) {
        mlModelSelect.addEventListener('change', updateModelDescription);
        updateModelDescription();
    }
});

// Add debugging function
function debugResponse(response) {
    console.log('Response status:', response.status);
    console.log('Response headers:', response.headers);
    return response;
}

// Updated prediction function with better error handling
async function makePrediction(event) {
    event.preventDefault(); // Prevent form submission

    try {
        // Get form values
        const ml_model = document.getElementById('ml_model').value;
        const forecast_period = document.getElementById('forecast_period').value;

        // Create request body
        const requestBody = {
            ml_model: ml_model,
            forecast_period: parseInt(forecast_period)
        };

        console.log('Sending request with data:', requestBody); // Debug log

        // Make API call
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody) // Stringify the data
        });

        // Check if response is ok
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }

        // Parse response
        const data = await response.json();
        console.log('Received response:', data); // Debug log

        // Handle successful prediction
        // Update your UI here with the prediction results

    } catch (error) {
        console.error('Error:', error);
        // Handle error - update UI to show error message
    }
}

// Add UI feedback function
function showMessage(message, type = 'info') {
    const messageDiv = document.getElementById('message-container') || createMessageContainer();
    messageDiv.textContent = message;
    messageDiv.className = `alert alert-${type} mt-3`;
    setTimeout(() => messageDiv.remove(), 5000);
}

// Create message container if it doesn't exist
function createMessageContainer() {
    const container = document.createElement('div');
    container.id = 'message-container';
    document.querySelector('form').appendChild(container);
    return container;
}

// Add event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    const predictButton = document.getElementById('predict-button');
    if (predictButton) {
        predictButton.addEventListener('click', (e) => {
            e.preventDefault();
            makePrediction();
        });
    }

    // Add form validation
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', makePrediction);
    }
});

// Form validation
function validateForm() {
    const ml_model = document.getElementById('ml_model').value;
    const forecast_period = document.getElementById('forecast_period').value;

    if (!ml_model) {
        showMessage('Please select a model', 'warning');
        return false;
    }

    if (!forecast_period || forecast_period < 1) {
        showMessage('Please enter a valid forecast period', 'warning');
        return false;
    }

    return true;
}
