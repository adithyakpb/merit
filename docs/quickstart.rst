Quickstart Guide
===============

This guide will help you get started with MERIT quickly. We'll cover the basic concepts and show you how to use MERIT for monitoring and evaluating AI systems.

Basic Concepts
-------------

MERIT is built around several core concepts:

- **Metrics**: Measurements of system performance and behavior
- **Evaluators**: Components that calculate metrics from data
- **Collectors**: Components that gather data from various sources
- **Storage**: Systems for persisting metrics and data
- **Reporting**: Tools for visualizing and analyzing results

Your First MERIT Project
-----------------------

Let's create a simple example that monitors an AI system:

.. code-block:: python

   from merit.metrics import BaseMetric
   from merit.monitoring import MonitoringService
   from merit.storage import FileStorage

   # Create a simple metric
   class ResponseTimeMetric(BaseMetric):
       name = "Response Time"
       description = "Measures the time taken for AI system responses"
       greater_is_better = False

       def __call__(self, response_time_ms):
           return {"value": response_time_ms, "unit": "ms"}

   # Set up monitoring
   storage = FileStorage("metrics_data")
   monitoring = MonitoringService(storage)

   # Record a metric
   metric = ResponseTimeMetric()
   result = metric(150)  # 150ms response time
   monitoring.record_metric("response_time", result)

Using the Command Line Interface
------------------------------

MERIT provides a command-line interface for common tasks:

.. code-block:: bash

   # Initialize a new MERIT project
   merit init

   # Run evaluations
   merit evaluate --config evaluation_config.json

   # Generate reports
   merit report --output report.html

   # Start monitoring dashboard
   merit dashboard

Integration with AI Systems
-------------------------

Here's how to integrate MERIT with your AI system:

.. code-block:: python

   from merit.monitoring import MonitoringService
   from merit.storage import FileStorage
   import time

   class AIMonitor:
       def __init__(self):
           self.storage = FileStorage("ai_metrics")
           self.monitoring = MonitoringService(self.storage)

       def monitor_response(self, prompt, response, response_time):
           # Record response time
           self.monitoring.record_metric("response_time", {
               "value": response_time,
               "prompt": prompt,
               "response": response
           })

       def monitor_accuracy(self, expected, actual):
           # Record accuracy
           accuracy = 1.0 if expected == actual else 0.0
           self.monitoring.record_metric("accuracy", {
               "value": accuracy,
               "expected": expected,
               "actual": actual
           })

   # Usage
   monitor = AIMonitor()
   
   # Simulate AI response
   start_time = time.time()
   response = "AI response here"
   response_time = (time.time() - start_time) * 1000
   
   monitor.monitor_response("Hello", response, response_time)
   monitor.monitor_accuracy("expected", "actual") 