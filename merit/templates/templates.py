"""
MERIT Templates Management

Enhanced template management system for monitoring dashboards, evaluation reports,
and configurable UI components that adapt to available metrics.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from ..metrics import list_metrics, get_metric_metadata, MetricContext, MetricCategory
from ..core.logging import get_logger

logger = get_logger(__name__)


class TemplateType(Enum):
    """Types of templates available in MERIT."""
    EVALUATION_REPORT = "evaluation_report"
    MONITORING_DASHBOARD = "monitoring_dashboard"
    RAG_MONITORING = "rag_monitoring"
    METRICS_CONFIG = "metrics_config"
    CUSTOM = "custom"


class UIContext(Enum):
    """UI contexts for different use cases."""
    MONITORING = "monitoring"
    EVALUATION = "evaluation"
    RAG_ANALYSIS = "rag_analysis"
    CONFIGURATION = "configuration"


class TemplateManager:
    """
    Manages templates for different UI contexts and automatically configures
    them based on available metrics and user preferences.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory containing template files
        """
        if templates_dir is None:
            templates_dir = os.path.dirname(__file__)
        
        self.templates_dir = Path(templates_dir)
        self.static_dir = self.templates_dir / "static"
        self.components_dir = self.static_dir / "components"
        
        # Ensure directories exist
        self.static_dir.mkdir(exist_ok=True)
        (self.static_dir / "css").mkdir(exist_ok=True)
        (self.static_dir / "js").mkdir(exist_ok=True)
        self.components_dir.mkdir(exist_ok=True)
        
        # Template registry
        self.templates = {}
        self.configurations = {}
        
        # Load default configurations
        self._load_default_configurations()
    
    def _load_default_configurations(self):
        """Load default template configurations."""
        self.configurations = {
            UIContext.MONITORING: {
                "metrics": list_metrics(context=MetricContext.MONITORING),
                "layout": "grid",
                "refresh_interval": 5000,  # 5 seconds
                "charts": ["time_series", "gauge", "counter"],
                "alerts": True,
                "real_time": True
            },
            UIContext.EVALUATION: {
                "metrics": list_metrics(context=MetricContext.EVALUATION),
                "layout": "report",
                "refresh_interval": None,  # Static
                "charts": ["bar", "radar", "distribution"],
                "alerts": False,
                "real_time": False
            },
            UIContext.RAG_ANALYSIS: {
                "metrics": [m for m in list_metrics() if "rag" in m.lower() or any(
                    keyword in m.lower() for keyword in ["correctness", "faithfulness", "relevance", "coherence", "fluency", "context"]
                )],
                "layout": "rag_focused",
                "refresh_interval": 10000,  # 10 seconds
                "charts": ["scatter", "heatmap", "correlation"],
                "alerts": True,
                "real_time": True
            },
            UIContext.CONFIGURATION: {
                "metrics": list_metrics(),  # All metrics
                "layout": "config_panel",
                "refresh_interval": None,
                "charts": ["preview"],
                "alerts": False,
                "real_time": False
            }
        }
    
    def get_template_config(self, context: UIContext, 
                           custom_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get template configuration for a specific context.
        
        Args:
            context: UI context
            custom_metrics: Optional list of custom metrics to include
            
        Returns:
            Template configuration dictionary
        """
        config = self.configurations.get(context, {}).copy()
        
        if custom_metrics:
            # Filter metrics to only include requested ones
            available_metrics = set(config.get("metrics", []))
            config["metrics"] = [m for m in custom_metrics if m in available_metrics]
        
        # Add metric metadata
        config["metric_metadata"] = {}
        for metric_name in config.get("metrics", []):
            try:
                config["metric_metadata"][metric_name] = get_metric_metadata(metric_name)
            except Exception as e:
                logger.warning(f"Could not get metadata for metric {metric_name}: {e}")
        
        # Add timestamp
        config["generated_at"] = datetime.now().isoformat()
        
        return config
    
    def generate_dashboard_config(self, 
                                 context: UIContext,
                                 metrics: Optional[List[str]] = None,
                                 layout_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a complete dashboard configuration.
        
        Args:
            context: UI context
            metrics: List of metrics to include
            layout_preferences: Custom layout preferences
            
        Returns:
            Complete dashboard configuration
        """
        base_config = self.get_template_config(context, metrics)
        
        # Apply layout preferences
        if layout_preferences:
            base_config.update(layout_preferences)
        
        # Generate widget configurations
        widgets = []
        for metric_name in base_config.get("metrics", []):
            widget_config = self._generate_widget_config(metric_name, context)
            if widget_config:
                widgets.append(widget_config)
        
        base_config["widgets"] = widgets
        
        # Generate layout grid
        base_config["layout_grid"] = self._generate_layout_grid(widgets, base_config.get("layout", "grid"))
        
        return base_config
    
    def _generate_widget_config(self, metric_name: str, context: UIContext) -> Optional[Dict[str, Any]]:
        """
        Generate widget configuration for a specific metric.
        
        Args:
            metric_name: Name of the metric
            context: UI context
            
        Returns:
            Widget configuration or None if metric not supported
        """
        try:
            metadata = get_metric_metadata(metric_name)
        except Exception:
            return None
        
        # Determine widget type based on metric category and context
        widget_type = self._determine_widget_type(metadata, context)
        
        widget_config = {
            "id": f"widget_{metric_name.lower().replace(' ', '_')}",
            "metric_name": metric_name,
            "type": widget_type,
            "title": metadata.get("name", metric_name),
            "description": metadata.get("description", ""),
            "category": metadata.get("category", MetricCategory.CUSTOM).name,
            "greater_is_better": metadata.get("greater_is_better", True),
            "size": self._determine_widget_size(widget_type),
            "refresh_interval": self._determine_refresh_interval(context, metadata),
            "chart_config": self._generate_chart_config(widget_type, metadata)
        }
        
        return widget_config
    
    def _determine_widget_type(self, metadata: Dict[str, Any], context: UIContext) -> str:
        """Determine the appropriate widget type for a metric."""
        category = metadata.get("category", MetricCategory.CUSTOM)
        metric_context = metadata.get("context", MetricContext.BOTH)
        
        # Context-specific widget types
        if context == UIContext.MONITORING:
            if category == MetricCategory.PERFORMANCE:
                return "gauge"
            elif category == MetricCategory.USAGE:
                return "counter"
            elif category == MetricCategory.COST:
                return "currency_display"
            else:
                return "time_series"
        
        elif context == UIContext.EVALUATION:
            return "metric_card"
        
        elif context == UIContext.RAG_ANALYSIS:
            if "context" in metadata.get("name", "").lower():
                return "heatmap"
            elif "faithfulness" in metadata.get("name", "").lower():
                return "correlation_chart"
            else:
                return "scatter_plot"
        
        return "basic_display"
    
    def _determine_widget_size(self, widget_type: str) -> Dict[str, int]:
        """Determine widget size based on type."""
        size_map = {
            "gauge": {"width": 2, "height": 2},
            "counter": {"width": 1, "height": 1},
            "currency_display": {"width": 1, "height": 1},
            "time_series": {"width": 4, "height": 2},
            "metric_card": {"width": 1, "height": 1},
            "heatmap": {"width": 3, "height": 3},
            "correlation_chart": {"width": 3, "height": 2},
            "scatter_plot": {"width": 2, "height": 2},
            "basic_display": {"width": 1, "height": 1}
        }
        
        return size_map.get(widget_type, {"width": 2, "height": 2})
    
    def _determine_refresh_interval(self, context: UIContext, metadata: Dict[str, Any]) -> Optional[int]:
        """Determine refresh interval for a widget."""
        base_interval = self.configurations.get(context, {}).get("refresh_interval")
        
        if base_interval is None:
            return None
        
        # Adjust based on metric category
        category = metadata.get("category", MetricCategory.CUSTOM)
        
        if category == MetricCategory.PERFORMANCE:
            return base_interval  # Real-time for performance metrics
        elif category == MetricCategory.COST:
            return base_interval * 6  # Less frequent for cost metrics
        else:
            return base_interval * 2  # Medium frequency for others
    
    def _generate_chart_config(self, widget_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart configuration for a widget."""
        base_config = {
            "responsive": True,
            "animation": True,
            "theme": "auto"  # Will adapt to light/dark theme
        }
        
        # Widget-specific configurations
        if widget_type == "gauge":
            base_config.update({
                "min": 0,
                "max": 1 if metadata.get("greater_is_better") is not None else 100,
                "thresholds": [0.3, 0.7, 0.9],
                "colors": ["#dc3545", "#ffc107", "#28a745", "#007bff"]
            })
        
        elif widget_type == "time_series":
            base_config.update({
                "time_window": "1h",
                "aggregation": "avg",
                "show_points": False,
                "fill": True
            })
        
        elif widget_type == "heatmap":
            base_config.update({
                "color_scale": "viridis",
                "show_scale": True
            })
        
        return base_config
    
    def _generate_layout_grid(self, widgets: List[Dict[str, Any]], layout_type: str) -> Dict[str, Any]:
        """Generate layout grid configuration."""
        if layout_type == "grid":
            # Auto-arrange widgets in a grid
            total_width = 12  # Bootstrap-style 12-column grid
            current_row = 0
            current_col = 0
            
            layout = []
            
            for widget in widgets:
                size = widget["size"]
                
                # Check if widget fits in current row
                if current_col + size["width"] > total_width:
                    current_row += 1
                    current_col = 0
                
                layout.append({
                    "widget_id": widget["id"],
                    "x": current_col,
                    "y": current_row,
                    "width": size["width"],
                    "height": size["height"]
                })
                
                current_col += size["width"]
            
            return {
                "type": "grid",
                "columns": total_width,
                "items": layout
            }
        
        elif layout_type == "rag_focused":
            # RAG-specific layout with dedicated sections
            return {
                "type": "sections",
                "sections": [
                    {
                        "title": "Document Retrieval",
                        "widgets": [w["id"] for w in widgets if "context" in w["metric_name"].lower()]
                    },
                    {
                        "title": "Generation Quality",
                        "widgets": [w["id"] for w in widgets if any(
                            keyword in w["metric_name"].lower() 
                            for keyword in ["correctness", "faithfulness", "relevance"]
                        )]
                    },
                    {
                        "title": "Language Quality",
                        "widgets": [w["id"] for w in widgets if any(
                            keyword in w["metric_name"].lower() 
                            for keyword in ["coherence", "fluency"]
                        )]
                    }
                ]
            }
        
        else:
            # Default layout
            return {
                "type": "list",
                "items": [{"widget_id": w["id"]} for w in widgets]
            }
    
    def save_configuration(self, context: UIContext, config: Dict[str, Any], 
                          name: Optional[str] = None) -> str:
        """
        Save a dashboard configuration.
        
        Args:
            context: UI context
            config: Configuration to save
            name: Optional name for the configuration
            
        Returns:
            Configuration ID
        """
        if name is None:
            name = f"{context.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config_file = self.templates_dir / f"config_{name}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Saved configuration '{name}' to {config_file}")
        return name
    
    def load_configuration(self, name: str) -> Dict[str, Any]:
        """
        Load a saved dashboard configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration dictionary
        """
        config_file = self.templates_dir / f"config_{name}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return config
    
    def list_configurations(self) -> List[str]:
        """List all saved configurations."""
        config_files = list(self.templates_dir.glob("config_*.json"))
        return [f.stem.replace("config_", "") for f in config_files]


# Global template manager instance
_template_manager = None

def get_template_manager() -> TemplateManager:
    """Get the global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager


def generate_monitoring_dashboard_config(metrics: Optional[List[str]] = None,
                                        layout_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate configuration for a monitoring dashboard.
    
    Args:
        metrics: List of metrics to include
        layout_preferences: Custom layout preferences
        
    Returns:
        Dashboard configuration
    """
    manager = get_template_manager()
    return manager.generate_dashboard_config(
        UIContext.MONITORING, 
        metrics, 
        layout_preferences
    )


def generate_evaluation_report_config(metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate configuration for an evaluation report.
    
    Args:
        metrics: List of metrics to include
        
    Returns:
        Report configuration
    """
    manager = get_template_manager()
    return manager.generate_dashboard_config(UIContext.EVALUATION, metrics)


def generate_rag_monitoring_config(metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate configuration for RAG monitoring dashboard.
    
    Args:
        metrics: List of metrics to include
        
    Returns:
        RAG monitoring configuration
    """
    manager = get_template_manager()
    return manager.generate_dashboard_config(UIContext.RAG_ANALYSIS, metrics)
