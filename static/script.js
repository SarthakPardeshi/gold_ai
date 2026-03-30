document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const btnSampleData = document.getElementById('btnSampleData');
    const btnLiveFetch = document.getElementById('btnLiveFetch');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultArea = document.getElementById('resultArea');
    const predictedPrice = document.getElementById('predictedPrice');
    const errorArea = document.getElementById('errorArea');
    const errorMessage = document.getElementById('errorMessage');
    let chartInstance = null;

    // Make Lucide icons work dynamically
    lucide.createIcons();

    if (btnLiveFetch) {
        btnLiveFetch.addEventListener('click', () => {
            hideUI();
            showLoading();
            
            fetch('/fetch_live')
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'success') {
                        sendPredictionRequest(data.features, data.dates);
                    } else {
                        throw new Error(data.message);
                    }
                })
                .catch(err => {
                    hideLoading();
                    showError("Live Fetch Failed: " + err.message);
                });
        });
    }

    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('dragover');
        }, false);
    });

    dropArea.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file);
    });

    function handleFile(file) {
        if (!file || !file.name.endsWith('.csv')) {
            showError("Please upload a valid CSV file.");
            return;
        }

        hideUI();
        showLoading();

        Papa.parse(file, {
            header: false,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {
                try {
                    const data = results.data;
                    
                    let valid7Cols = [];
                    let rawPrices = [];
                    
                    for (let row of data) {
                        if (!Array.isArray(row)) continue;
                        let nums = [];
                        for (let val of row) {
                            if (val === null || val === '') continue;
                            let n = Number(val);
                            if (!isNaN(n)) nums.push(n);
                        }
                        
                        if (nums.length >= 7) {
                            valid7Cols.push(nums.slice(0, 7));
                        }
                        if (nums.length > 0) {
                            rawPrices.push(nums[0]); // Typically 'Close' price
                        }
                    }

                    // If strictly 7 columns already exist
                    if (valid7Cols.length >= 60) {
                        sendPredictionRequest(valid7Cols.slice(-60));
                        return;
                    }

                    // Otherwise, trigger dynamic Feature Engineering Pipeline
                    if (rawPrices.length < 60) {
                        throw new Error(`Insufficient data. Found only ${rawPrices.length} usable market days. The neural network sequence requires at least 60.`);
                    }

                    // Advanced Feature Engineering in JS
                    const USD_INR = 83.5; // Benchmark Exchange Rate
                    let pricesINR = rawPrices.map(p => p * USD_INR);
                    let engineeredFeatures = [];
                    
                    let ema10 = pricesINR[0];
                    let k = 2 / 11; // Standard EMA alpha
                    
                    for (let i = 0; i < rawPrices.length; i++) {
                        let p = pricesINR[i];
                        
                        // Moving Average 10
                        let ma10 = p;
                        if (i >= 9) {
                            let sum = 0;
                            for (let j = 0; j < 10; j++) sum += pricesINR[i - j];
                            ma10 = sum / 10;
                        }
                        
                        // Moving Average 50
                        let ma50 = p;
                        if (i >= 49) {
                            let sum = 0;
                            for (let j = 0; j < 50; j++) sum += pricesINR[i - j];
                            ma50 = sum / 50;
                        }
                        
                        // EMA 10
                        if (i > 0) ema10 = (p - ema10) * k + ema10;
                        
                        // Volatility (Std Dev 10)
                        let vol = 0;
                        if (i >= 9) {
                            let mean = ma10;
                            let sumSq = 0;
                            for (let j = 0; j < 10; j++) sumSq += Math.pow(pricesINR[i - j] - mean, 2);
                            vol = Math.sqrt(sumSq / 9); // sample std
                        }
                        
                        // Custom RSI 14
                        let rsi = 50;
                        if (i >= 14) {
                            let gains = 0, losses = 0;
                            for (let j = 0; j < 14; j++) {
                                let diff = pricesINR[i - j] - pricesINR[i - j - 1];
                                if (diff > 0) gains += diff;
                                else losses -= diff;
                            }
                            let avgGain = gains / 14;
                            let avgLoss = losses / 14;
                            if (avgLoss === 0) rsi = 100;
                            else rsi = 100 - (100 / (1 + (avgGain / avgLoss)));
                        }
                        
                        engineeredFeatures.push([
                            rawPrices[i], USD_INR, ma10, ma50, ema10, vol, rsi
                        ]);
                    }

                    const numericData = engineeredFeatures.slice(-60);
                    sendPredictionRequest(numericData);
                } catch (error) {
                    hideLoading();
                    showError(error.message);
                }
            },
            error: function(error) {
                hideLoading();
                showError("Failed to parse CSV: " + error.message);
            }
        });
    }

    btnSampleData.addEventListener('click', () => {
        hideUI();
        showLoading();
        
        let rawPrices = [];
        let baseVal = 2000; // Gold USD realistic price

        // Generate 110 days to feed into our engine
        for (let i = 0; i < 110; i++) {
            baseVal += (Math.random() - 0.5) * 15; 
            rawPrices.push(baseVal);
        }

        // Advanced Feature Engineering in JS
        const USD_INR = 83.5; 
        let pricesINR = rawPrices.map(p => p * USD_INR);
        let engineeredFeatures = [];
        
        let ema10 = pricesINR[0];
        let k = 2 / 11; 
        
        for (let i = 0; i < rawPrices.length; i++) {
            let p = pricesINR[i];
            
            let ma10 = p;
            if (i >= 9) {
                let sum = 0;
                for (let j = 0; j < 10; j++) sum += pricesINR[i - j];
                ma10 = sum / 10;
            }
            
            let ma50 = p;
            if (i >= 49) {
                let sum = 0;
                for (let j = 0; j < 50; j++) sum += pricesINR[i - j];
                ma50 = sum / 50;
            }
            
            if (i > 0) ema10 = (p - ema10) * k + ema10;
            
            let vol = 0;
            if (i >= 9) {
                let mean = ma10;
                let sumSq = 0;
                for (let j = 0; j < 10; j++) sumSq += Math.pow(pricesINR[i - j] - mean, 2);
                vol = Math.sqrt(sumSq / 9); 
            }
            
            let rsi = 50;
            if (i >= 14) {
                let gains = 0, losses = 0;
                for (let j = 0; j < 14; j++) {
                    let diff = pricesINR[i - j] - pricesINR[i - j - 1];
                    if (diff > 0) gains += diff;
                    else losses -= diff;
                }
                let avgGain = gains / 14;
                let avgLoss = losses / 14;
                if (avgLoss === 0) rsi = 100;
                else rsi = 100 - (100 / (1 + (avgGain / avgLoss)));
            }
            
            engineeredFeatures.push([
                rawPrices[i], USD_INR, ma10, ma50, ema10, vol, rsi
            ]);
        }

        setTimeout(() => {
            sendPredictionRequest(engineeredFeatures.slice(-60));
        }, 800);
    });

    async function sendPredictionRequest(features, datesArray = null) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ features: features })
            });

            const result = await response.json();

            if (result.status === 'success') {
                hideLoading();
                showResult(result.predicted_gold_price_inr);
                
                // Extract close prices. Gold=USD/oz (index 0), USD_INR (index 1). We graph standard Indian pricing (INR per 10 grams)
                // 1 Troy Ounce = 31.1034768 grams, so 10g = 1 oz / 3.11034768
                const inrHistoryPrices = features.map(f => (f[0] * f[1]) / 3.11034768);
                
                // Generate fallback labels if real dates aren't available
                let chartLabels = datesArray;
                if (!chartLabels) {
                    chartLabels = Array.from({length: features.length}, (_, i) => `Day ${i - features.length + 1}`);
                }
                
                renderChart(inrHistoryPrices, result.predicted_gold_price_inr, chartLabels);
            } else {
                throw new Error(result.message || 'Unknown server error');
            }
        } catch (error) {
            hideLoading();
            showError("API Error: " + error.message);
        }
    }

    function showLoading() {
        loadingIndicator.classList.remove('hidden');
    }

    function hideLoading() {
        loadingIndicator.classList.add('hidden');
    }

    function hideUI() {
        resultArea.classList.add('hidden');
        errorArea.classList.add('hidden');
    }

    function showResult(price) {
        // Animate counting up for extra wow factor
        resultArea.classList.remove('hidden');
        predictedPrice.innerText = "0.00";
        
        const target = parseFloat(price);
        const duration = 1500;
        const frames = 60;
        const step = target / frames;
        let current = 0;
        
        const counter = setInterval(() => {
            current += step;
            if (current >= target) {
                clearInterval(counter);
                current = target;
            }
            predictedPrice.innerText = current.toLocaleString('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
        }, duration / frames);
    }

    function showError(msg) {
        errorMessage.innerText = msg;
        errorArea.classList.remove('hidden');
    }

    function renderChart(historicalPrices, predictedValue, labels) {
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        if (chartInstance) {
            chartInstance.destroy();
        }
        
        const chartData = [...historicalPrices];
        
        labels.push('Next Day');
        const predData = new Array(historicalPrices.length).fill(null);
        predData[historicalPrices.length - 1] = historicalPrices[historicalPrices.length - 1]; // connect the line
        predData.push(predictedValue);
        
        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Historical Price (₹ / 10g)',
                        data: chartData,
                        borderColor: '#FFD700',
                        backgroundColor: 'rgba(255, 215, 0, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'AI Prediction',
                        data: predData,
                        borderColor: '#4caf50',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0,
                        pointRadius: 4,
                        pointBackgroundColor: '#4caf50'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f0f0f5', font: { family: 'Outfit' } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) {
                                    label += '₹' + context.parsed.y.toLocaleString('en-IN', {maximumFractionDigits: 0});
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#a0a0b0', maxTicksLimit: 12 }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { 
                            color: '#a0a0b0',
                            callback: function(value) {
                                return '₹' + value.toLocaleString('en-IN');
                            }
                        },
                        title: {
                            display: true,
                            text: 'Price (₹ per 10 grams)',
                            color: '#FFD700'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
            }
        });
    }
});
