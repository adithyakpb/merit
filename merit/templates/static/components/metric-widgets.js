/**
 * MERIT Metric Widgets
 * 
 * Factory for creating different types of metric widgets that can be used
 * in both monitoring and evaluation contexts.
 */

class MetricWidgetFactory {
    constructor(metricsEngine) {
        this.metricsEngine = metricsEngine;
        this.widgets = new Map();
        this.chartInstances = new Map();
        
        // Load Chart.js if not already loaded
        this.ensureChartJS();
    }
    
    ensureChartJS() {
        if (typeof Chart === 'undefined') {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
            document.head.appendChild(script);
        }
    }
    
    createWidget(config, container) {
        const { type, metric_name, title, description } = config;
        
        // Create widget wrapper
        const widget = this.createWidgetWrapper(config, container);
        
        // Create widget content based on type
        let widgetInstance;
        switch (type) {
            case 'gauge':
                widgetInstance = this.createGaugeWidget(config, widget);
                break;
            case 'counter':
                widgetInstance = this.createCounterWidget(config, widget);
                break;
            case 'currency_display':
                widgetInstance = this.createCurrencyWidget(config, widget);
                break;
            case 'time_series':
                widgetInstance = this.createTimeSeriesWidget(config, widget);
                break;
            case 'metric_card':
                widgetInstance = this.createMetricCardWidget(config, widget);
                break;
            case 'heatmap':
                widgetInstance = this.createHeatmapWidget(config, widget);
                break;
            case 'correlation_chart':
                widgetInstance = this.createCorrelationWidget(config, widget);
                break;
            case 'scatter_plot':
                widgetInstance = this.createScatterWidget(config, widget);
                break;
            default:
                widgetInstance = this.createBasicWidget(config, widget);
        }
        
        // Store widget instance
        this.widgets.set(config.id, widgetInstance);
        
        // Subscribe to metric updates
        if (this.metricsEngine && metric_name) {
            this.metricsEngine.subscribe(metric_name, (data) => {
                widgetInstance.update(data);
            });
        }
        
        return widgetInstance;
    }
    
    createWidgetWrapper(config, container) {
        const { id, title, description, size, type } = config;
        
        const widget = document.createElement('div');
        widget.className = `widget widget-${type}`;
        widget.id = id;
        widget.style.gridColumn = `span ${size.width}`;
        widget.style.gridRow = `span ${size.height}`;
        
        // Add real-time indicator for monitoring widgets
        if (config.refresh_interval) {
            const indicator = document.createElement('div');
            indicator.className = 'real-time-indicator';
            widget.appendChild(indicator);
        }
        
        // Widget header
        const header = document.createElement('div');
        header.className = 'widget-header';
        
        const titleElement = document.createElement('h3');
        titleElement.className = 'widget-title';
        titleElement.textContent = title;
        
        const actions = document.createElement('div');
        actions.className = 'widget-actions';
        
        // Add fullscreen button
        const fullscreenBtn = document.createElement('button');
        fullscreenBtn.className = 'widget-action';
        fullscreenBtn.innerHTML = '⛶';
        fullscreenBtn.title = 'Fullscreen';
        fullscreenBtn.onclick = () => this.toggleFullscreen(widget);
        actions.appendChild(fullscreenBtn);
        
        header.appendChild(titleElement);
        header.appendChild(actions);
        widget.appendChild(header);
        
        // Widget content container
        const content = document.createElement('div');
        content.className = 'widget-content';
        widget.appendChild(content);
        
        // Add to container
        container.appendChild(widget);
        
        return widget;
    }
    
    createGaugeWidget(config, widget) {
        const content = widget.querySelector('.widget-content');
        const { chart_config } = config;
        
        // Create gauge container
        const gaugeContainer = document.createElement('div');
        gaugeContainer.className = 'gauge-container';
        
        const canvas = document.createElement('canvas');
        canvas.width = 120;
        canvas.height = 120;
        gaugeContainer.appendChild(canvas);
        
        const valueDisplay = document.createElement('div');
        valueDisplay.className = 'gauge-value';
        valueDisplay.textContent = '0';
        gaugeContainer.appendChild(valueDisplay);
        
        const label = document.createElement('div');
        label.className = 'gauge-label';
        label.textContent = config.title;
        
        content.appendChild(gaugeContainer);
        content.appendChild(label);
        
        // Create gauge chart
        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [0, 100],
                    backgroundColor: [chart_config.colors[0], '#e9ecef'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
                cutout: '70%',
                rotation: -90,
                circumference: 180,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                }
            }
        });
        
        this.chartInstances.set(config.id, chart);
        
        return {
            update: (data) => {
                const value = data.current;
                const percentage = (value / chart_config.max) * 100;
                
                valueDisplay.textContent = typeof value === 'number' ? value.toFixed(2) : value;
                
                chart.data.datasets[0].data = [percentage, 100 - percentage];
                
                // Update color based on thresholds
                const colorIndex = this.getThresholdColorIndex(value, chart_config.thresholds);
                chart.data.datasets[0].backgroundColor[0] = chart_config.colors[colorIndex];
                
                chart.update('none');
            },
            destroy: () => {
                chart.destroy();
                this.chartInstances.delete(config.id);
            }
        };
    }
    
    createCounterWidget(config, widget) {
        const content = widget.querySelector('.widget-content');
        
        const counterValue = document.createElement('div');
        counterValue.className = 'counter-value';
        counterValue.textContent = '0';
        
        const counterLabel = document.createElement('div');
        counterLabel.className = 'counter-label';
        counterLabel.textContent = config.title;
        
        content.appendChild(counterValue);
        content.appendChild(counterLabel);
        
        return {
            update: (data) => {
                const value = data.current;
                const aggregated = data.aggregated;
                
                // Animate value change
                counterValue.classList.add('updating');
                setTimeout(() => counterValue.classList.remove('updating'), 300);
                
                if (typeof value === 'number') {
                    counterValue.textContent = this.formatNumber(value);
                } else {
                    counterValue.textContent = value;
                }
                
                // Show trend if available
                if (aggregated && aggregated.trend) {
                    counterLabel.className = `counter-label trend-${aggregated.trend}`;
                }
            },
            destroy: () => {}
        };
    }
    
    createCurrencyWidget(config, widget) {
        const content = widget.querySelector('.widget-content');
        
        const currencyValue = document.createElement('div');
        currencyValue.className = 'currency-value';
        currencyValue.textContent = '0.00';
        
        const currencyLabel = document.createElement('div');
        currencyLabel.className = 'currency-label';
        currencyLabel.textContent = config.title;
        
        const currencyChange = document.createElement('div');
        currencyChange.className = 'currency-change';
        
        content.appendChild(currencyValue);
        content.appendChild(currencyLabel);
        content.appendChild(currencyChange);
        
        let previousValue = 0;
        
        return {
            update: (data) => {
                const value = data.current;
                const change = value - previousValue;
                
                currencyValue.textContent = this.formatCurrency(value);
                
                if (change !== 0) {
                    currencyChange.textContent = this.formatCurrency(Math.abs(change));
                    currencyChange.className = `currency-change ${change > 0 ? 'positive' : 'negative'}`;
                    currencyChange.style.display = 'block';
                } else {
                    currencyChange.style.display = 'none';
                }
                
                previousValue = value;
            },
            destroy: () => {}
        };
    }
    
    createTimeSeriesWidget(config, widget) {
        const content = widget.querySelector('.widget-content');
        
        // Add time range controls
        const controls = document.createElement('div');
        controls.className = 'time-series-controls';
        
        const timeRangeSelector = document.createElement('div');
        timeRangeSelector.className = 'time-range-selector';
        
        const ranges = ['5m', '15m', '1h', '6h', '24h'];
        ranges.forEach(range => {
            const btn = document.createElement('button');
            btn.className = 'time-range-btn';
            btn.textContent = range;
            btn.onclick = () => {
                document.querySelectorAll('.time-range-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                // Request new data for this time range
                if (this.metricsEngine) {
                    this.metricsEngine.requestMetricHistory(config.metric_name, range);
                }
            };
            if (range === '1h') btn.classList.add('active');
            timeRangeSelector.appendChild(btn);
        });
        
        controls.appendChild(timeRangeSelector);
        content.appendChild(controls);
        
        // Chart container
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        
        const canvas = document.createElement('canvas');
        chartContainer.appendChild(canvas);
        content.appendChild(chartContainer);
        
        // Create time series chart
        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: config.title,
                    data: [],
                    borderColor: getComputedStyle(document.documentElement).getPropertyValue('--primary-color'),
                    backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--primary-color') + '20',
                    fill: config.chart_config.fill,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            displayFormats: {
                                minute: 'HH:mm',
                                hour: 'HH:mm'
                            }
                        }
                    },
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
        
        this.chartInstances.set(config.id, chart);
        
        return {
            update: (data) => {
                if (data.history && data.history.length > 0) {
                    const labels = data.history.map(item => item.timestamp);
                    const values = data.history.map(item => item.value);
                    
                    chart.data.labels = labels;
                    chart.data.datasets[0].data = values;
                    chart.update('none');
                }
            },
            destroy: () => {
                chart.destroy();
                this.chartInstances.delete(config.id);
            }
        };
    }
    
    createMetricCardWidget(config, widget) {
        const content = widget.querySelector('.widget-content');
        
        const card = document.createElement('div');
        card.className = 'metric-card';
        
        const value = document.createElement('div');
        value.className = 'metric-value';
        value.textContent = '0';
        
        const label = document.createElement('div');
        label.className = 'metric-label';
        label.textContent = config.title;
        
        const change = document.createElement('div');
        change.className = 'metric-change';
        
        card.appendChild(value);
        card.appendChild(label);
        card.appendChild(change);
        content.appendChild(card);
        
        return {
            update: (data) => {
                const currentValue = data.current;
                value.textContent = typeof currentValue === 'number' ? currentValue.toFixed(3) : currentValue;
                
                if (data.aggregated && data.aggregated.trend) {
                    change.textContent = `Trend: ${data.aggregated.trend}`;
                    change.className = `metric-change ${data.aggregated.trend}`;
                }
            },
            destroy: () => {}
        };
    }
    
    createHeatmapWidget(config, widget) {
        const content = widget.querySelector('.widget-content');
        
        // For now, create a placeholder for heatmap
        // In a real implementation, you'd use a library like D3.js or Plotly
        const heatmapContainer = document.createElement('div');
        heatmapContainer.className = 'heatmap-placeholder';
        heatmapContainer.innerHTML = `
            <div style="text-align: center; padding: 20px;">
                <h4>${config.title} Heatmap</h4>
                <p>Heatmap visualization would be rendered here</p>
                <div style="background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00); height: 20px; margin: 10px 0;"></div>
                <small>Low → High</small>
            </div>
        `;
        
        content.appendChild(heatmapContainer);
        
        return {
            update: (data) => {
                // Update heatmap with new data
                console.log('Updating heatmap with:', data);
            },
            destroy: () => {}
        };
    }
    
    createCorrelationWidget(config, widget) {
        const content = widget.querySelector('.widget-content');
        
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        
        const canvas = document.createElement('canvas');
        chartContainer.appendChild(canvas);
        content.appendChild(chartContainer);
        
        // Create scatter plot for correlation
        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: config.title,
                    data: [],
                    backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--primary-color'),
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Faithfulness' } },
                    y: { title: { display: true, text: 'Relevance' } }
                }
            }
        });
        
        this.chartInstances.set(config.id, chart);
        
        return {
            update: (data) => {
                // Generate sample correlation data
                const correlationData = [];
                for (let i = 0; i < 20; i++) {
                    correlationData.push({
                        x: Math.random() * 0.4 + 0.6,
                        y: Math.random() * 0.4 + 0.6
                    });
                }
                
                chart.data.datasets[0].data = correlationData;
                chart.update('none');
            },
            destroy: () => {
                chart.destroy();
                this.chartInstances.delete(config.id);
            }
        };
    }
    
    createScatterWidget(config, widget) {
        const content = widget.querySelector('.widget-content');
        
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        
        const canvas = document.createElement('canvas');
        chartContainer.appendChild(canvas);
        content.appendChild(chartContainer);
        
        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: config.title,
                    data: [],
                    backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--primary-color'),
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
        
        this.chartInstances.set(config.id, chart);
        
        return {
            update: (data) => {
                // Update scatter plot
                chart.data.datasets[0].data = [{
                    x: Math.random() * 100,
                    y: data.current * 100
                }];
                chart.update('none');
            },
            destroy: () => {
                chart.destroy();
                this.chartInstances.delete(config.id);
            }
        };
    }
    
    createBasicWidget(config, widget) {
        const content = widget.querySelector('.widget-content');
        
        const basicDisplay = document.createElement('div');
        basicDisplay.className = 'basic-display';
        basicDisplay.innerHTML = `
            <h4>${config.title}</h4>
            <div class="basic-value">0</div>
            <div class="basic-description">${config.description}</div>
        `;
        
        content.appendChild(basicDisplay);
        
        return {
            update: (data) => {
                const valueElement = basicDisplay.querySelector('.basic-value');
                valueElement.textContent = data.current;
            },
            destroy: () => {}
        };
    }
    
    // Utility methods
    getThresholdColorIndex(value, thresholds) {
        for (let i = 0; i < thresholds.length; i++) {
            if (value <= thresholds[i]) {
                return i;
            }
        }
        return thresholds.length;
    }
    
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }
    
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 4
        }).format(amount);
    }
    
    toggleFullscreen(widget) {
        if (widget.classList.contains('widget-fullscreen')) {
            widget.classList.remove('widget-fullscreen');
        } else {
            widget.classList.add('widget-fullscreen');
        }
        
        // Trigger chart resize
        const widgetId = widget.id;
        const chart = this.chartInstances.get(widgetId);
        if (chart) {
            setTimeout(() => chart.resize(), 100);
        }
    }
    
    destroyWidget(widgetId) {
        const widget = this.widgets.get(widgetId);
        if (widget) {
            widget.destroy();
            this.widgets.delete(widgetId);
        }
    }
    
    destroyAllWidgets() {
        for (const [id, widget] of this.widgets.entries()) {
            widget.destroy();
        }
        this.widgets.clear();
        this.chartInstances.clear();
    }
}

// Export for use in other modules
window.MetricWidgetFactory = MetricWidgetFactory;
