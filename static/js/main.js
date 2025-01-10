// Model descriptions
const modelDescriptions = {
    'random_forest': '* Ensemble learning method using multiple decision trees. Best for balanced performance and handling non-linear patterns.',
    'gradient_boost': '* Sequential ensemble method that builds trees to correct previous errors. High accuracy with proper tuning.',
    'xgboost': '* Advanced implementation of gradient boosting with better regularization and parallel processing.',
    'lightgbm': '* Gradient boosting framework using leaf-wise tree growth. Fast training and memory efficient.',
    'svr': '* Kernel-based method that maps data to higher dimensions. Effective for non-linear relationships.',
    'neural_network': '* Deep learning model with multiple layers. Automatically learns complex patterns from data.'
};

document.addEventListener('DOMContentLoaded', function () {
    console.log('Application initialized');
    
    const form = document.querySelector('#prediction-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }

    const mlModelSelect = document.getElementById('ml_model');
    if (mlModelSelect) {
        mlModelSelect.addEventListener('change', updateModelDescription);
        updateModelDescription();
    }

    const applyFiltersBtn = document.getElementById('apply-filters');
    if (applyFiltersBtn) {
        applyFiltersBtn.addEventListener('click', handleFilters);
    }

    const exportCsvBtn = document.getElementById('export-csv');
    const exportJsonBtn = document.getElementById('export-json');
    if (exportCsvBtn) exportCsvBtn.addEventListener('click', () => exportData('csv'));
    if (exportJsonBtn) exportJsonBtn.addEventListener('click', () => exportData('json'));

    // Visualization type selector
    const vizButtons = document.querySelectorAll('.viz-type-selector button');
    vizButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            
            vizButtons.forEach(btn => btn.classList.remove('active'));
            e.target.classList.add('active');
            
            const vizType = e.target.dataset.viz;
            console.log(`Visualization button clicked: ${vizType}`);
            
            updateVisualizationType(vizType);
        });
    });

    // Real-time updates toggle
    const updateToggle = document.getElementById('enable-updates');
    let updateInterval;
    updateToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            updateInterval = setInterval(fetchLatestData, 30000); // Update every 30 seconds
            updateLastUpdateTime();
        } else {
            clearInterval(updateInterval);
        }
    });
});

async function handleFormSubmit(e) {
    e.preventDefault();
    console.log("Form submitted");

    if (!validateForm()) {
        return;
    }

    // Show loading overlay
    document.querySelector('.loading-overlay').style.display = 'flex';

    try {
        const formData = new FormData(e.target);
        const jsonData = {};
        
        // Convert form values to numbers where appropriate
        formData.forEach((value, key) => {
            if (['traffic', 'marketing', 'advertising', 'social'].includes(key)) {
                jsonData[key] = parseFloat(value) || 0;
            } else {
                jsonData[key] = value;
            }
        });

        console.log("Data being sent to /predict:", jsonData);

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(jsonData),
        });

        const data = await response.json();
        console.log("Prediction result:", data);

        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }

        if (data.success) {
            updateDashboard(data);
        } else {
            showError(data.error || 'An error occurred during prediction');
        }
    } catch (error) {
        console.error("Error:", error);
        showError(error.message);
    } finally {
        document.querySelector('.loading-overlay').style.display = 'none';
    }
}

function updateDashboard(data) {
    try {
        // Update prediction display
        const predictedSales = document.getElementById('predicted-sales');
        if (predictedSales) {
            predictedSales.textContent = `$${parseFloat(data.prediction).toFixed(2)}`;
        }

        const accuracyScore = document.getElementById('accuracy-score');
        if (accuracyScore) {
            accuracyScore.textContent = `${(parseFloat(data.accuracy_score) * 100).toFixed(2)}%`;
        }

        const confidenceScore = document.getElementById('confidence-score');
        if (confidenceScore) {
            confidenceScore.textContent = `${(parseFloat(data.confidence_score) * 100).toFixed(2)}%`;
        }

        // Update visualization if available
        if (data.plot && data.plot.data && data.plot.layout) {
            Plotly.newPlot('visualization-container', data.plot.data, data.plot.layout);
        } else {
            console.warn('No plot data available');
        }
    } catch (error) {
        console.error("Error updating dashboard:", error);
        showError("Error updating the dashboard display");
    }
}

function showError(message) {
    showMessage(message, 'danger');
}

function showMessage(message, type = 'info') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `alert alert-${type} mt-3`;
    messageDiv.textContent = message;
    
    // Find a suitable container for the message
    const container = document.querySelector('.visualization-container').parentElement;
    container.insertBefore(messageDiv, container.firstChild);
    
    // Remove the message after 3 seconds
    setTimeout(() => {
        messageDiv.remove();
    }, 3000);
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
    const traffic = parseFloat(document.getElementById('traffic')?.value);
    const marketing = parseFloat(document.getElementById('marketing')?.value);
    const advertising = parseFloat(document.getElementById('advertising')?.value);
    const social = parseFloat(document.getElementById('social')?.value);

    if (!mlModel) {
        showError('Please select a model');
        return false;
    }

    // Check if values are valid numbers and greater than or equal to 0
    if (isNaN(traffic) || traffic < 0 ||
        isNaN(marketing) || marketing < 0 ||
        isNaN(advertising) || advertising < 0 ||
        isNaN(social) || social < 0) {
        showError('Please enter valid positive numbers for all required fields');
        return false;
    }

    return true;
}

function handleFilters() {
    const startDate = document.getElementById('filter-start-date').value;
    const endDate = document.getElementById('filter-end-date').value;
    const selectedMetrics = Array.from(document.getElementById('metrics-select').selectedOptions)
        .map(option => option.value);

    if (!startDate || !endDate) {
        showMessage('Please select both start and end dates', 'warning');
        return;
    }

    // Send filter request to server
    fetch('/apply_filters', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            startDate,
            endDate,
            metrics: selectedMetrics
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateVisualization(data);
            showMessage('Filters applied successfully', 'success');
        } else {
            showError(data.error);
        }
    })
    .catch(error => {
        showError('Error applying filters: ' + error.message);
    });
}

function exportData(format) {
    const startDate = document.getElementById('filter-start-date').value;
    const endDate = document.getElementById('filter-end-date').value;
    const includeAnalysis = document.getElementById('include-analysis').checked;

    fetch('/export_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            format,
            startDate,
            endDate,
            includeAnalysis
        })
    })
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sales_data_${format}.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    })
    .catch(error => {
        showError('Error exporting data: ' + error.message);
    });
}

function updateVisualizationType(type) {
    console.log(`Updating visualization to: ${type}`);
    
    fetch('/update_visualization', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ type })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            Plotly.newPlot('visualization-container', 
                data.plot.data, 
                data.plot.layout,
                {responsive: true}
            );
            showMessage('Visualization updated successfully', 'success');
        } else {
            throw new Error(data.error || 'Failed to update visualization');
        }
    })
    .catch(error => {
        console.error('Error updating visualization:', error);
        showError('Error updating visualization: ' + error.message);
    });
}

function updateLastUpdateTime() {
    const timeElement = document.querySelector('.last-update-time');
    timeElement.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
}

function fetchLatestData() {
    fetch('/get_latest_data')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateDashboard(data);
                updateLastUpdateTime();
            }
        })
        .catch(error => console.error('Error fetching latest data:', error));
}

