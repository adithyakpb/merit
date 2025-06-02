/**
 * MERIT Real-Time Metrics Engine
 * 
 * Handles real-time metric updates, WebSocket connections, and data processing
 * for the monitoring dashboard.
 */

class RealTimeMetricsEngine {
    constructor(config = {}) {
        this.config = {
            websocketUrl: config.websocketUrl || 'ws://localhost:8000/ws/metrics',
            reconnectInterval: config.reconnectInterval || 5000,
            maxReconnectAttempts: config.maxReconnectAttempts || 10,
            bufferSize: config.bufferSize || 1000,
            updateInterval: config.updateInterval || 1000,
            ...config
        };
        
        // Internal state
        this.websocket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.metrics = new Map();
        this.subscribers = new Map();
        this.dataBuffer = new Map();
        this.lastUpdate = new Map();
        
        // Bind methods
        this.handleMessage = this.handleMessage.bind(this);
        this.handleOpen = this.handleOpen.bind(this);
        this.handleClose = this.handleClose.bind(this);
        this.handleError = this.handleError.bind(this);
        
        // Initialize
        this.init();
    }
    
    async init() {
        console.log('Initializing MERIT Real-Time Metrics Engine...');
        
        // Connect to WebSocket
        this.connect();
        
        // Start update loop
        this.startUpdateLoop();
        
        // Set up page visibility handling
        this.setupVisibilityHandling();
    }
    
    connect() {
        try {
            console.log(`Connecting to WebSocket: ${this.config.websocketUrl}`);
            
            this.websocket = new WebSocket(this.config.websocketUrl);
            this.websocket.onopen = this.handleOpen;
            this.websocket.onmessage = this.handleMessage;
            this.websocket.onclose = this.handleClose;
            this.websocket.onerror = this.handleError;
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.scheduleReconnect();
        }
    }
    
    handleOpen(event) {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // Notify subscribers about connection status
        this.notifyConnectionStatus('connected');
        
        // Request initial data
        this.requestInitialData();
    }
    
    handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            this.processMessage(data);
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
        }
    }
    
    handleClose(event) {
        console.log('WebSocket disconnected:', event.code, event.reason);
        this.isConnected = false;
        
        // Notify subscribers about connection status
        this.notifyConnectionStatus('disconnected');
        
        // Schedule reconnection if not a clean close
        if (event.code !== 1000) {
            this.scheduleReconnect();
        }
    }
    
    handleError(error) {
        console.error('WebSocket error:', error);
        this.notifyConnectionStatus('error');
    }
    
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.notifyConnectionStatus('failed');
            return;
        }
        
        this.reconnectAttempts++;
        console.log(`Scheduling reconnection attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts}`);
        
        this.notifyConnectionStatus('reconnecting');
        
        setTimeout(() => {
            this.connect();
        }, this.config.reconnectInterval);
    }
    
    processMessage(data) {
        const { type, payload } = data;
        
        switch (type) {
            case 'metric_update':
                this.handleMetricUpdate(payload);
                break;
                
            case 'interaction_data':
                this.handleInteractionData(payload);
                break;
                
            case 'alert':
                this.handleAlert(payload);
                break;
                
            case 'config_update':
                this.handleConfigUpdate(payload);
                break;
                
            default:
                console.warn('Unknown message type:', type);
        }
    }
    
    handleMetricUpdate(payload) {
        const { metric_name, value, timestamp, metadata } = payload;
        
        // Store metric data
        if (!this.dataBuffer.has(metric_name)) {
            this.dataBuffer.set(metric_name, []);
        }
        
        const buffer = this.dataBuffer.get(metric_name);
        buffer.push({
            value,
            timestamp: new Date(timestamp),
            metadata
        });
        
        // Maintain buffer size
        if (buffer.length > this.config.bufferSize) {
            buffer.shift();
        }
        
        // Update last update time
        this.lastUpdate.set(metric_name, new Date());
        
        // Notify subscribers
        this.notifySubscribers(metric_name, {
            current: value,
            history: buffer,
            metadata
        });
    }
    
    handleInteractionData(payload) {
        // Process raw interaction data and calculate metrics
        this.processInteraction(payload);
    }
    
    handleAlert(payload) {
        // Handle alert notifications
        this.notifySubscribers('alerts', payload);
    }
    
    handleConfigUpdate(payload) {
        // Handle configuration updates
        this.notifySubscribers('config', payload);
    }
    
    processInteraction(interaction) {
        // Calculate metrics from interaction data
        const metrics = this.calculateMetricsFromInteraction(interaction);
        
        // Update each calculated metric
        for (const [metricName, value] of Object.entries(metrics)) {
            this.handleMetricUpdate({
                metric_name: metricName,
                value: value,
                timestamp: new Date().toISOString(),
                metadata: { source: 'interaction' }
            });
        }
    }
    
    calculateMetricsFromInteraction(interaction) {
        const metrics = {};
        
        // Request count
        metrics['Request Volume'] = 1;
        
        // Latency
        if (interaction.response && interaction.response.latency) {
            metrics['Latency'] = interaction.response.latency;
        }
        
        // Token usage
        if (interaction.response && interaction.response.tokens) {
            const tokens = interaction.response.tokens;
            metrics['Token Volume'] = tokens.total_tokens || (tokens.input_tokens + tokens.output_tokens);
        }
        
        // Error rate
        const isError = interaction.response && interaction.response.status !== 'success';
        metrics['Error Rate'] = isError ? 1 : 0;
        
        // Cost estimation (simplified)
        if (interaction.response && interaction.response.tokens && interaction.model) {
            const cost = this.estimateCost(interaction.response.tokens, interaction.model);
            if (cost !== null) {
                metrics['Cost Estimate'] = cost;
            }
        }
        
        return metrics;
    }
    
    estimateCost(tokens, model) {
        // Simplified cost estimation
        const pricing = {
            'gpt-4': { input: 0.00003, output: 0.00006 },
            'gpt-3.5-turbo': { input: 0.0000015, output: 0.000002 },
            'claude-2': { input: 0.00001102, output: 0.00003268 }
        };
        
        const modelPricing = pricing[model] || pricing['gpt-3.5-turbo'];
        const inputCost = (tokens.input_tokens || 0) * modelPricing.input;
        const outputCost = (tokens.output_tokens || 0) * modelPricing.output;
        
        return inputCost + outputCost;
    }
    
    subscribe(metricName, callback) {
        if (!this.subscribers.has(metricName)) {
            this.subscribers.set(metricName, new Set());
        }
        
        this.subscribers.get(metricName).add(callback);
        
        // Send current data if available
        if (this.dataBuffer.has(metricName)) {
            const buffer = this.dataBuffer.get(metricName);
            if (buffer.length > 0) {
                callback({
                    current: buffer[buffer.length - 1].value,
                    history: buffer,
                    metadata: buffer[buffer.length - 1].metadata
                });
            }
        }
        
        // Return unsubscribe function
        return () => {
            const subscribers = this.subscribers.get(metricName);
            if (subscribers) {
                subscribers.delete(callback);
            }
        };
    }
    
    notifySubscribers(metricName, data) {
        const subscribers = this.subscribers.get(metricName);
        if (subscribers) {
            subscribers.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error('Error in subscriber callback:', error);
                }
            });
        }
    }
    
    notifyConnectionStatus(status) {
        this.notifySubscribers('connection', { status, timestamp: new Date() });
    }
    
    requestInitialData() {
        if (this.isConnected) {
            this.websocket.send(JSON.stringify({
                type: 'request_initial_data',
                payload: {}
            }));
        }
    }
    
    requestMetricHistory(metricName, timeRange = '1h') {
        if (this.isConnected) {
            this.websocket.send(JSON.stringify({
                type: 'request_metric_history',
                payload: { metric_name: metricName, time_range: timeRange }
            }));
        }
    }
    
    startUpdateLoop() {
        setInterval(() => {
            this.updateAggregatedMetrics();
        }, this.config.updateInterval);
    }
    
    updateAggregatedMetrics() {
        // Calculate aggregated metrics from buffered data
        for (const [metricName, buffer] of this.dataBuffer.entries()) {
            if (buffer.length === 0) continue;
            
            const aggregated = this.calculateAggregatedMetrics(buffer);
            this.notifySubscribers(`${metricName}_aggregated`, aggregated);
        }
    }
    
    calculateAggregatedMetrics(buffer) {
        const values = buffer.map(item => item.value);
        
        return {
            count: values.length,
            sum: values.reduce((a, b) => a + b, 0),
            avg: values.reduce((a, b) => a + b, 0) / values.length,
            min: Math.min(...values),
            max: Math.max(...values),
            latest: values[values.length - 1],
            trend: this.calculateTrend(values)
        };
    }
    
    calculateTrend(values) {
        if (values.length < 2) return 'stable';
        
        const recent = values.slice(-5); // Last 5 values
        const older = values.slice(-10, -5); // Previous 5 values
        
        if (recent.length === 0 || older.length === 0) return 'stable';
        
        const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
        const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
        
        const change = (recentAvg - olderAvg) / olderAvg;
        
        if (change > 0.05) return 'up';
        if (change < -0.05) return 'down';
        return 'stable';
    }
    
    setupVisibilityHandling() {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Page is hidden, reduce update frequency
                console.log('Page hidden, reducing update frequency');
            } else {
                // Page is visible, resume normal updates
                console.log('Page visible, resuming normal updates');
                if (!this.isConnected) {
                    this.connect();
                }
            }
        });
    }
    
    getMetricData(metricName) {
        return this.dataBuffer.get(metricName) || [];
    }
    
    getConnectionStatus() {
        return {
            connected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts,
            lastUpdate: this.lastUpdate
        };
    }
    
    disconnect() {
        if (this.websocket) {
            this.websocket.close(1000, 'Client disconnect');
        }
    }
    
    // Simulate data for demo purposes
    startSimulation() {
        console.log('Starting metrics simulation...');
        
        const simulateMetric = (metricName, generator) => {
            setInterval(() => {
                const value = generator();
                this.handleMetricUpdate({
                    metric_name: metricName,
                    value: value,
                    timestamp: new Date().toISOString(),
                    metadata: { source: 'simulation' }
                });
            }, 2000 + Math.random() * 3000); // Random interval between 2-5 seconds
        };
        
        // Simulate different metrics
        simulateMetric('Request Volume', () => Math.floor(Math.random() * 100) + 50);
        simulateMetric('Latency', () => Math.random() * 2 + 0.5);
        simulateMetric('Token Volume', () => Math.floor(Math.random() * 1000) + 500);
        simulateMetric('Error Rate', () => Math.random() * 0.1);
        simulateMetric('Cost Estimate', () => Math.random() * 0.05 + 0.01);
        
        // Simulate RAG metrics
        simulateMetric('Correctness', () => Math.random() * 0.3 + 0.7);
        simulateMetric('Faithfulness', () => Math.random() * 0.2 + 0.8);
        simulateMetric('Relevance', () => Math.random() * 0.25 + 0.75);
        simulateMetric('ContextPrecision', () => Math.random() * 0.4 + 0.6);
    }
}

// Export for use in other modules
window.RealTimeMetricsEngine = RealTimeMetricsEngine;
