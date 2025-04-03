# Report Templates

This guide explains how to create and use custom report templates in MERIT. Report templates allow you to customize the presentation of evaluation results to suit your specific needs and preferences.

## Why Create Custom Report Templates?

You might want to create custom report templates for several reasons:

- **Specialized Reporting**: Create reports tailored to specific evaluation scenarios
- **Custom Visualizations**: Include custom visualizations of evaluation results
- **Organizational Standards**: Adhere to your organization's reporting standards
- **Focused Analysis**: Highlight specific aspects of evaluation results
- **Integration**: Integrate evaluation results with other systems or tools

## Report Types in MERIT

MERIT supports several types of reports:

- **Text Reports**: Plain text reports for simple presentation
- **Markdown Reports**: Markdown-formatted reports for better readability
- **HTML Reports**: Interactive HTML reports with visualizations
- **JSON Reports**: Structured JSON reports for programmatic access
- **CSV Reports**: Tabular CSV reports for spreadsheet analysis

## Creating a Basic Report Template

To create a custom report template, you need to define a template function that takes an `EvaluationReport` object and returns a formatted report:

```python
from merit.core.models import EvaluationReport
from typing import Dict, Any

def custom_text_report(report: EvaluationReport) -> str:
    """
    Generate a custom text report from an evaluation report.
    
    Args:
        report: The evaluation report to format.
        
    Returns:
        str: The formatted report.
    """
    # Create the report header
    header = f"Evaluation Report\n"
    header += f"================\n\n"
    
    # Add metadata
    metadata = report.metadata or {}
    header += f"Evaluator: {metadata.get('evaluator', 'Unknown')}\n"
    header += f"Metrics: {', '.join(report.metrics)}\n"
    header += f"Number of samples: {len(report.results)}\n\n"
    
    # Add overall scores
    overall_scores = report.get_overall_scores()
    header += f"Overall Scores\n"
    header += f"-------------\n"
    for metric, score in overall_scores.items():
        header += f"{metric}: {score:.2f}\n"
    header += "\n"
    
    # Add sample results
    sample_results = "Sample Results\n"
    sample_results += "-------------\n"
    for i, result in enumerate(report.results):
        sample_results += f"Sample {i+1}:\n"
        sample_results += f"Input: {result.input}\n"
        sample_results += f"Model answer: {result.model_answer}\n"
        sample_results += f"Scores:\n"
        for metric, score in result.scores.items():
            sample_results += f"  {metric}: {score:.2f}\n"
        
        # Add explanations if available
        if result.explanations:
            sample_results += f"Explanations:\n"
            for metric, explanation in result.explanations.items():
                if explanation:
                    sample_results += f"  {metric}: {explanation}\n"
        
        # Add errors if available
        if result.errors:
            sample_results += f"Errors:\n"
            for metric, error in result.errors.items():
                if error:
                    sample_results += f"  {metric}: {error}\n"
        
        sample_results += "\n"
    
    # Combine the report sections
    report_text = header + sample_results
    
    return report_text
```

## Creating a Markdown Report Template

You can create a Markdown report template for better readability:

```python
def custom_markdown_report(report: EvaluationReport) -> str:
    """
    Generate a custom Markdown report from an evaluation report.
    
    Args:
        report: The evaluation report to format.
        
    Returns:
        str: The formatted report.
    """
    # Create the report header
    header = f"# Evaluation Report\n\n"
    
    # Add metadata
    metadata = report.metadata or {}
    header += f"**Evaluator:** {metadata.get('evaluator', 'Unknown')}  \n"
    header += f"**Metrics:** {', '.join(report.metrics)}  \n"
    header += f"**Number of samples:** {len(report.results)}  \n\n"
    
    # Add overall scores
    overall_scores = report.get_overall_scores()
    header += f"## Overall Scores\n\n"
    header += f"| Metric | Score |\n"
    header += f"|--------|-------|\n"
    for metric, score in overall_scores.items():
        header += f"| {metric} | {score:.2f} |\n"
    header += "\n"
    
    # Add sample results
    sample_results = "## Sample Results\n\n"
    for i, result in enumerate(report.results):
        sample_results += f"### Sample {i+1}\n\n"
        sample_results += f"**Input:** {result.input}  \n"
        sample_results += f"**Model answer:** {result.model_answer}  \n\n"
        sample_results += f"**Scores:**  \n"
        for metric, score in result.scores.items():
            sample_results += f"- {metric}: {score:.2f}  \n"
        
        # Add explanations if available
        if result.explanations:
            sample_results += f"\n**Explanations:**  \n"
            for metric, explanation in result.explanations.items():
                if explanation:
                    sample_results += f"- {metric}: {explanation}  \n"
        
        # Add errors if available
        if result.errors:
            sample_results += f"\n**Errors:**  \n"
            for metric, error in result.errors.items():
                if error:
                    sample_results += f"- {metric}: {error}  \n"
        
        sample_results += "\n"
    
    # Combine the report sections
    report_text = header + sample_results
    
    return report_text
```

## Creating an HTML Report Template

You can create an HTML report template with visualizations:

```python
def custom_html_report(report: EvaluationReport) -> str:
    """
    Generate a custom HTML report from an evaluation report.
    
    Args:
        report: The evaluation report to format.
        
    Returns:
        str: The formatted report.
    """
    # Create the HTML header
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            h1, h2, h3 {
                color: #333;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .sample {
                margin-bottom: 30px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .chart {
                width: 600px;
                height: 400px;
                margin: 20px 0;
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
    """
    
    # Add the report header
    metadata = report.metadata or {}
    html += f"<h1>Evaluation Report</h1>"
    html += f"<p><strong>Evaluator:</strong> {metadata.get('evaluator', 'Unknown')}</p>"
    html += f"<p><strong>Metrics:</strong> {', '.join(report.metrics)}</p>"
    html += f"<p><strong>Number of samples:</strong> {len(report.results)}</p>"
    
    # Add overall scores
    overall_scores = report.get_overall_scores()
    html += f"<h2>Overall Scores</h2>"
    html += f"<table>"
    html += f"<tr><th>Metric</th><th>Score</th></tr>"
    for metric, score in overall_scores.items():
        html += f"<tr><td>{metric}</td><td>{score:.2f}</td></tr>"
    html += f"</table>"
    
    # Add a chart for overall scores
    html += f"""
    <div class="chart">
        <canvas id="overallScoresChart"></canvas>
    </div>
    <script>
        var ctx = document.getElementById('overallScoresChart').getContext('2d');
        var chart = new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {list(overall_scores.keys())},
                datasets: [{{
                    label: 'Overall Scores',
                    data: {list(overall_scores.values())},
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1
                    }}
                }}
            }}
        }});
    </script>
    """
    
    # Add sample results
    html += f"<h2>Sample Results</h2>"
    for i, result in enumerate(report.results):
        html += f'<div class="sample">'
        html += f"<h3>Sample {i+1}</h3>"
        html += f"<p><strong>Input:</strong> {result.input}</p>"
        html += f"<p><strong>Model answer:</strong> {result.model_answer}</p>"
        
        # Add scores
        html += f"<h4>Scores</h4>"
        html += f"<table>"
        html += f"<tr><th>Metric</th><th>Score</th></tr>"
        for metric, score in result.scores.items():
            html += f"<tr><td>{metric}</td><td>{score:.2f}</td></tr>"
        html += f"</table>"
        
        # Add a chart for sample scores
        html += f"""
        <div class="chart">
            <canvas id="sampleScoresChart{i}"></canvas>
        </div>
        <script>
            var ctx = document.getElementById('sampleScoresChart{i}').getContext('2d');
            var chart = new Chart(ctx, {{
                type: 'radar',
                data: {{
                    labels: {list(result.scores.keys())},
                    datasets: [{{
                        label: 'Scores',
                        data: {list(result.scores.values())},
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    scales: {{
                        r: {{
                            beginAtZero: true,
                            max: 1
                        }}
                    }}
                }}
            }});
        </script>
        """
        
        # Add explanations if available
        if result.explanations:
            html += f"<h4>Explanations</h4>"
            html += f"<ul>"
            for metric, explanation in result.explanations.items():
                if explanation:
                    html += f"<li><strong>{metric}:</strong> {explanation}</li>"
            html += f"</ul>"
        
        # Add errors if available
        if result.errors:
            html += f"<h4>Errors</h4>"
            html += f"<ul>"
            for metric, error in result.errors.items():
                if error:
                    html += f"<li><strong>{metric}:</strong> {error}</li>"
            html += f"</ul>"
        
        html += f'</div>'
    
    # Add the HTML footer
    html += """
    </body>
    </html>
    """
    
    return html
```

## Creating a JSON Report Template

You can create a JSON report template for programmatic access:

```python
import json

def custom_json_report(report: EvaluationReport) -> str:
    """
    Generate a custom JSON report from an evaluation report.
    
    Args:
        report: The evaluation report to format.
        
    Returns:
        str: The formatted report.
    """
    # Create a dictionary for the report
    report_dict = {
        "metadata": report.metadata or {},
        "metrics": report.metrics,
        "overall_scores": report.get_overall_scores(),
        "samples": []
    }
    
    # Add sample results
    for i, result in enumerate(report.results):
        sample_dict = {
            "id": i + 1,
            "input": result.input,
            "model_answer": result.model_answer,
            "scores": result.scores,
            "explanations": result.explanations or {},
            "errors": result.errors or {}
        }
        
        report_dict["samples"].append(sample_dict)
    
    # Convert the dictionary to JSON
    report_json = json.dumps(report_dict, indent=2)
    
    return report_json
```

## Creating a CSV Report Template

You can create a CSV report template for spreadsheet analysis:

```python
import csv
import io

def custom_csv_report(report: EvaluationReport) -> str:
    """
    Generate a custom CSV report from an evaluation report.
    
    Args:
        report: The evaluation report to format.
        
    Returns:
        str: The formatted report.
    """
    # Create a string buffer for the CSV data
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write the header row
    header = ["Sample ID", "Input", "Model Answer"]
    for metric in report.metrics:
        header.append(f"{metric} Score")
    writer.writerow(header)
    
    # Write the sample rows
    for i, result in enumerate(report.results):
        row = [i + 1, result.input, result.model_answer]
        for metric in report.metrics:
            row.append(result.scores.get(metric, ""))
        writer.writerow(row)
    
    # Get the CSV data as a string
    csv_data = output.getvalue()
    output.close()
    
    return csv_data
```

## Using Custom Report Templates

Once you've created a custom report template, you can use it to format evaluation reports:

```python
from merit.evaluation import evaluate_rag
from merit.core.models import TestSet
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient

# Load a test set and knowledge base
test_set = TestSet.load("my_test_set.json")
knowledge_base = KnowledgeBase.load("my_knowledge_base.json")

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Define an answer function
def get_answer(query):
    # Your RAG system implementation here
    return "This is the answer to the query."

# Evaluate the RAG system
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)

# Generate reports using custom templates
text_report = custom_text_report(report)
markdown_report = custom_markdown_report(report)
html_report = custom_html_report(report)
json_report = custom_json_report(report)
csv_report = custom_csv_report(report)

# Save the reports to files
with open("report.txt", "w") as f:
    f.write(text_report)

with open("report.md", "w") as f:
    f.write(markdown_report)

with open("report.html", "w") as f:
    f.write(html_report)

with open("report.json", "w") as f:
    f.write(json_report)

with open("report.csv", "w") as f:
    f.write(csv_report)
```

## Advanced Report Templates

### Customizing Report Sections

You can customize the sections included in your reports:

```python
def custom_report_with_sections(report: EvaluationReport, sections=None) -> str:
    """
    Generate a custom report with specific sections.
    
    Args:
        report: The evaluation report to format.
        sections: The sections to include in the report.
            Options: "metadata", "overall_scores", "sample_results", "analysis"
            
    Returns:
        str: The formatted report.
    """
    # Set default sections if not provided
    if sections is None:
        sections = ["metadata", "overall_scores", "sample_results"]
    
    # Create the report header
    text = f"Evaluation Report\n"
    text += f"================\n\n"
    
    # Add metadata section
    if "metadata" in sections:
        metadata = report.metadata or {}
        text += f"Metadata\n"
        text += f"--------\n"
        text += f"Evaluator: {metadata.get('evaluator', 'Unknown')}\n"
        text += f"Metrics: {', '.join(report.metrics)}\n"
        text += f"Number of samples: {len(report.results)}\n\n"
    
    # Add overall scores section
    if "overall_scores" in sections:
        overall_scores = report.get_overall_scores()
        text += f"Overall Scores\n"
        text += f"-------------\n"
        for metric, score in overall_scores.items():
            text += f"{metric}: {score:.2f}\n"
        text += "\n"
    
    # Add sample results section
    if "sample_results" in sections:
        text += f"Sample Results\n"
        text += f"-------------\n"
        for i, result in enumerate(report.results):
            text += f"Sample {i+1}:\n"
            text += f"Input: {result.input}\n"
            text += f"Model answer: {result.model_answer}\n"
            text += f"Scores:\n"
            for metric, score in result.scores.items():
                text += f"  {metric}: {score:.2f}\n"
            
            # Add explanations if available
            if result.explanations:
                text += f"Explanations:\n"
                for metric, explanation in result.explanations.items():
                    if explanation:
                        text += f"  {metric}: {explanation}\n"
            
            text += "\n"
    
    # Add analysis section
    if "analysis" in sections:
        text += f"Analysis\n"
        text += f"--------\n"
        
        # Calculate average scores per metric
        metric_scores = {}
        for metric in report.metrics:
            scores = [result.scores.get(metric, 0) for result in report.results]
            avg_score = sum(scores) / len(scores)
            metric_scores[metric] = avg_score
        
        # Identify the best and worst metrics
        best_metric = max(metric_scores.items(), key=lambda x: x[1])
        worst_metric = min(metric_scores.items(), key=lambda x: x[1])
        
        text += f"Best performing metric: {best_metric[0]} ({best_metric[1]:.2f})\n"
        text += f"Worst performing metric: {worst_metric[0]} ({worst_metric[1]:.2f})\n\n"
        
        # Identify the best and worst samples
        sample_scores = []
        for i, result in enumerate(report.results):
            avg_score = sum(result.scores.values()) / len(result.scores)
            sample_scores.append((i, avg_score))
        
        best_sample = max(sample_scores, key=lambda x: x[1])
        worst_sample = min(sample_scores, key=lambda x: x[1])
        
        text += f"Best performing sample: Sample {best_sample[0]+1} ({best_sample[1]:.2f})\n"
        text += f"Worst performing sample: Sample {worst_sample[0]+1} ({worst_sample[1]:.2f})\n"
    
    return text
```

### Adding Custom Visualizations

You can add custom visualizations to your HTML reports:

```python
def custom_html_report_with_visualizations(report: EvaluationReport) -> str:
    """
    Generate a custom HTML report with visualizations.
    
    Args:
        report: The evaluation report to format.
        
    Returns:
        str: The formatted report.
    """
    # Create the HTML header with Chart.js
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report</title>
        <style>
            /* CSS styles */
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
    """
    
    # Add the report header
    # ...
    
    # Add a bar chart for overall scores
    overall_scores = report.get_overall_scores()
    html += f"""
    <div class="chart">
        <h2>Overall Scores</h2>
        <canvas id="overallScoresChart"></canvas>
    </div>
    <script>
        var ctx = document.getElementById('overallScoresChart').getContext('2d');
        var chart = new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {list(overall_scores.keys())},
                datasets: [{{
                    label: 'Overall Scores',
                    data: {list(overall_scores.values())},
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1
                    }}
                }}
            }}
        }});
    </script>
    """
    
    # Add a radar chart for metric comparison
    html += f"""
    <div class="chart">
        <h2>Metric Comparison</h2>
        <canvas id="metricComparisonChart"></canvas>
    </div>
    <script>
        var ctx = document.getElementById('metricComparisonChart').getContext('2d');
        var chart = new Chart(ctx, {{
            type: 'radar',
            data: {{
                labels: {list(overall_scores.keys())},
                datasets: [{{
                    label: 'Overall Scores',
                    data: {list(overall_scores.values())},
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 1
                    }}
                }}
            }}
        }});
    </script>
    """
    
    # Add a line chart for sample scores
    html += f"""
    <div class="chart">
        <h2>Sample Scores</h2>
        <canvas id="sampleScoresChart"></canvas>
    </div>
    <script>
        var ctx = document.getElementById('sampleScoresChart').getContext('2d');
        var chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {list(range(1, len(report.results) + 1))},
                datasets: [
    """
    
    # Add a dataset for each metric
    for i, metric in enumerate(report.metrics):
        scores = [result.scores.get(metric, 0) for result in report.results]
        html += f"""
                    {{
                        label: '{metric}',
                        data: {scores},
                        borderColor: 'hsl({i * 360 / len(report.metrics)}, 70%, 50%)',
                        tension: 0.1
                    }},
        """
    
    html += f"""
                ]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1
                    }}
                }}
            }}
        }});
    </script>
    """
    
    # Add sample results
    # ...
    
    # Add the HTML footer
    html += """
    </body>
    </html>
    """
    
    return html
```

### Creating Interactive Reports

You can create interactive reports with JavaScript:

```python
def custom_interactive_html_report(report: EvaluationReport) -> str:
    """
    Generate a custom interactive HTML report.
    
    Args:
        report: The evaluation report to format.
        
    Returns:
        str: The formatted report.
    """
    # Create the HTML header with Chart.js and interactive features
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Evaluation Report</title>
        <style>
            /* CSS styles */
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            function filterSamples() {
                const minScore = parseFloat(document.getElementById('minScore').value);
                const maxScore = parseFloat(document.getElementById('maxScore').value);
                const metric = document.getElementById('metricFilter').value;
                
                const samples = document.getElementsByClassName('sample');
                for (let i = 0; i < samples.length; i++) {
                    const sample = samples[i];
                    const score = parseFloat(sample.getAttribute('data-' + metric));
                    
                    if (score >= minScore && score <= maxScore) {
                        sample.style.display = 'block';
                    } else {
                        sample.style.display = 'none';
                    }
                }
                
                updateFilteredStats();
            }
            
            function updateFilteredStats() {
                const samples = document.getElementsByClassName('sample');
                let visibleCount = 0;
                let totalScore = 0;
                
                for (let i = 0; i < samples.length; i++) {
                    if (samples[i].style.display !== 'none') {
                        visibleCount++;
                        const metric = document.getElementById('metricFilter').value;
                        totalScore += parseFloat(samples[i].getAttribute('data-' + metric));
                    }
                }
                
                const avgScore = visibleCount > 0 ? totalScore / visibleCount : 0;
                document.getElementById('filteredCount').textContent = visibleCount;
                document.getElementById('filteredAvg').textContent = avgScore.toFixed(2);
            }
            
            function sortSamples() {
                const metric = document.getElementById('sortMetric').value;
                const direction = document.getElementById('sortDirection').value;
                
                const samplesContainer = document.getElementById('samplesContainer');
                const samples = Array.from(document.getElementsByClassName('sample'));
                
                samples.sort((a, b) => {
                    const scoreA = parseFloat(a.getAttribute('data-' + metric));
                    const scoreB = parseFloat(b.getAttribute('data-' + metric));
                    
                    return direction === 'asc' ? scoreA - scoreB : scoreB - scoreA;
                });
                
                // Remove all samples
                while (samplesContainer.firstChild) {
                    samplesContainer.removeChild(samplesContainer.firstChild);
                }
                
                // Add sorted samples
                for (const sample of samples) {
                    samplesContainer.appendChild(sample);
                }
            }
        </script>
    </head>
    <body>
    """
    
    # Add the report header
    # ...
    
    # Add filter controls
    html += f"""
    <div class="controls">
        <h2>Filter Samples</h2>
        <div>
            <label for="metricFilter">Metric:</label>
            <select id="metricFilter" onchange="filterSamples()">
    """
    
    for metric in report.metrics:
        html += f'<option value="{metric}">{metric}</option>'
    
    html += f"""
            </select>
        </div>
        <div>
            <label for="minScore">Min Score:</label>
            <input type="range" id="minScore" min="0" max="1" step="0.1" value="0" oninput="filterSamples()">
            <span id="minScoreValue">0</span>
        </div>
        <div>
            <label for="maxScore">Max Score:</label>
            <input type="range" id="maxScore" min="0" max="1" step="0.1" value="1" oninput="filterSamples()">
            <span id="maxScoreValue">1</span>
        </div>
        <div>
            <p>Filtered samples: <span id="filteredCount">{len(report.results)}</span></p>
            <p>Average score: <span id="filteredAvg">0.00</span></p>
        </div>
    </div>
    
    <div class="controls">
        <h2>Sort Samples</h2>
        <div>
            <label for="sortMetric">Metric:</label>
            <select id="sortMetric">
    """
    
    for metric in report.metrics:
        html += f'<option value="{metric}">{metric}</option>'
    
    html += f"""
            </select>
        </div>
        <div>
            <label for="sortDirection">Direction:</label>
            <select id="sortDirection">
                <option value="desc">Highest to Lowest</option>
                <option value="asc">Lowest to Highest</option>
            </select>
        </div>
        <button onclick="sortSamples()">Sort</button>
    </div>
    """
    
    # Add sample results with data attributes for filtering and sorting
    html += f'<div id="samplesContainer">'
    for i, result in enumerate(report.results):
        # Add data attributes for each metric
        data_attrs = ""
        for metric, score in result.scores.items():
            data_attrs += f' data-{metric}="{score}"'
        
        html += f'<div class="sample"{data_attrs}>'
        # Sample content
        html += f"<h3>Sample {i+1}</h3>"
        # ...
        html += f'</div>'
    html += f'</div>'
    
    # Add JavaScript to initialize the filter values
    html += """
    <script>
        document.getElementById('minScoreValue').textContent = document.getElementById('minScore').value;
        document.getElementById('maxScoreValue').textContent = document.getElementById('maxScore').value;
        
        document.getElementById('minScore').addEventListener('input', function() {
            document.getElementById('minScoreValue').textContent = this.value;
        });
        
        document.getElementById('maxScore').addEventListener('input', function() {
            document.getElementById('maxScoreValue').textContent = this.value;
        });
        
        updateFilteredStats();
    </script>
    """
    
    # Add the HTML footer
    html += """
    </body>
    </html>
    """
    
    return html
```

## Best Practices for Report Templates

When creating custom report templates, follow these best practices:

### 1. Include Essential Information

Ensure your reports include essential information
