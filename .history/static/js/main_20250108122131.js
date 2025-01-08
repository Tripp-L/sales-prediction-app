// Model descriptions and other constants
const modelDescriptions = {
    'random_forest': '* Ensemble learning method using multiple decision trees. Best for balanced performance and handling non-linear patterns.',
    'gradient_boost': '* Sequential ensemble method that builds trees to correct previous errors. High accuracy with proper tuning.',
    'xgboost': '* Advanced implementation of gradient boosting with better regularization and parallel processing.',
    'lightgbm': '* Gradient boosting framework using leaf-wise tree growth. Fast training and memory efficient.',
    'svr': '* Kernel-based method that maps data to higher dimensions. Effective for non-linear relationships.',
    'neural_network': '* Deep learning model with multiple layers. Automatically learns complex patterns from data.'
};

document.addEventListener('DOMContentLoaded', function() {
    // Initialize form elements
    const form = document.querySelector('#prediction-form');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const errorMessage = document.getElementById('error-message');
    const dateInput = document.getElementById('date');
    const mlModelSelect = document.getElementById('ml_model');
    
    // Set default date
    if (dateInput) {
        dateInput.valueAsDate = new Date();
    }
    
    // Model description updates
    if (mlModelSelect) {
        mlModelSelect.addEventListener('change', updateModelDescription);
        updateModelDescription();
    }
    
    // Form submission
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    
    // Initialize visualization controls
    initializeVisualizationControls();
    
    // Initialize WebSocket connection if needed
    initializeWebSocket();
    
    // Helper functions
    function updateModelDescription() {
        const selectedModel = mlModelSelect.value;
        const descriptionElement = document.getElementById('model-description');
        if (descriptionElement) {
            descriptionElement.textContent = modelDescriptions[selectedModel] || '';
        }
    }
    
    function showLoading() {
        if (loadingOverlay) loadingOverlay.style.display = 'flex';
        if (errorMessage) errorMessage.style.display = 'none';
    }
    
    function hideLoading() {
        if (loadingOverlay) loadingOverlay.style.display = 'none';
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
    
    async function handleFormSubmit(e) {
        e.preventDefault();
        showLoading();
        
        try {
            const formData = new FormData(form);
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || 'Prediction failed');
            }
            
            const data = await response.json();
            updateDashboard(data);
            
        } catch (error) {
            showError(error.message);
            console.error('Error:', error);
        } finally {
            hideLoading();
        }
    }
    
    function updateDashboard(data) {
        // Update metrics
        document.getElementById('predicted-sales').textContent = 
            `$${parseFloat(data.prediction).toFixed(2)}`;
        document.getElementById('accuracy-score').textContent = 
            `${(data.accuracy * 100).toFixed(1)}%`;
        document.getElementById('confidence-score').textContent = 
            `${(data.confidence * 100).toFixed(1)}%`;
        
        // Update visualizations
        if (data.plot) {
            Plotly.newPlot(
                'visualization-container',
                data.plot.data,
                data.plot.layout,
                {responsive: true}
            );
        }
    }
    
    function initializeVisualizationControls() {
        // Add event listeners for visualization controls
        const vizButtons = document.querySelectorAll('.viz-type-selector button');
        vizButtons.forEach(button => {
            button.addEventListener('click', function() {
                updateVisualization(this.dataset.viz);
            });
        });
    }
    
    function initializeWebSocket() {
        // Check if Socket.IO is available
        if (typeof io !== 'undefined') {
            const socket = io();
            
            socket.on('connect', () => {
                console.log('WebSocket connected');
            });
            
            socket.on('data_update', (data) => {
                console.log('Received data update:', data);
                // Handle real-time updates
            });
        } else {
            console.log('Socket.IO not available - real-time updates disabled');
        }
    }
});
