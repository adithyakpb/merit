#!/usr/bin/env python3
"""
MERIT Dashboard Features Demo

This script demonstrates the key features of the MERIT monitoring dashboard system:
1. Extensible template management
2. Context-aware metric configuration
3. Real-time widget generation
4. RAG metrics integration
"""

import json
from datetime import datetime
from merit.templates.templates import (
    get_template_manager,
    generate_monitoring_dashboard_config,
    generate_rag_monitoring_config,
    UIContext
)
from merit.metrics import list_metrics, MetricContext, get_metric_metadata


def demo_extensible_template_system():
    """Demonstrate the extensible template management system."""
    print("🎯 MERIT Dashboard Features Demo")
    print("=" * 50)
    
    # Get template manager
    manager = get_template_manager()
    
    print("\n1. 🔧 EXTENSIBLE TEMPLATE SYSTEM")
    print("-" * 30)
    
    # Show available contexts
    print("Available UI Contexts:")
    for context in UIContext:
        config = manager.get_template_config(context)
        print(f"  • {context.value}: {len(config.get('metrics', []))} metrics")
    
    # Show how easy it is to add new metrics
    print("\n📊 Automatic Metric Detection:")
    monitoring_metrics = list_metrics(context=MetricContext.MONITORING)
    evaluation_metrics = list_metrics(context=MetricContext.EVALUATION)
    
    print(f"  Monitoring metrics: {len(monitoring_metrics)}")
    print(f"  Evaluation metrics: {len(evaluation_metrics)}")
    print("  ✅ New metrics are automatically detected and configured!")


def demo_context_aware_configuration():
    """Demonstrate context-aware metric configuration."""
    print("\n2. 🎨 CONTEXT-AWARE CONFIGURATION")
    print("-" * 35)
    
    # Generate different dashboard configs
    monitoring_config = generate_monitoring_dashboard_config()
    rag_config = generate_rag_monitoring_config()
    
    print("Monitoring Dashboard:")
    print(f"  • Widgets: {len(monitoring_config.get('widgets', []))}")
    print(f"  • Layout: {monitoring_config.get('layout', 'N/A')}")
    print(f"  • Real-time: {monitoring_config.get('real_time', False)}")
    
    print("\nRAG Monitoring Dashboard:")
    print(f"  • Widgets: {len(rag_config.get('widgets', []))}")
    print(f"  • Layout: {rag_config.get('layout', 'N/A')}")
    print(f"  • RAG-specific: {rag_config.get('layout') == 'rag_focused'}")


def demo_widget_generation():
    """Demonstrate automatic widget generation."""
    print("\n3. 🔮 AUTOMATIC WIDGET GENERATION")
    print("-" * 35)
    
    manager = get_template_manager()
    
    # Show how widgets are automatically configured
    sample_metrics = ["Request Volume", "Latency", "Token Volume", "Correctness"]
    
    for metric_name in sample_metrics:
        try:
            metadata = get_metric_metadata(metric_name)
            widget_config = manager._generate_widget_config(metric_name, UIContext.MONITORING)
            
            if widget_config:
                print(f"\n{metric_name}:")
                print(f"  • Widget Type: {widget_config['type']}")
                print(f"  • Size: {widget_config['size']['width']}x{widget_config['size']['height']}")
                print(f"  • Category: {widget_config['category']}")
                print(f"  • Refresh: {widget_config.get('refresh_interval', 'Static')}ms")
        except Exception as e:
            print(f"  ⚠️ {metric_name}: Not available ({e})")


def demo_rag_integration():
    """Demonstrate RAG metrics integration."""
    print("\n4. 🔍 RAG METRICS INTEGRATION")
    print("-" * 30)
    
    # Show RAG-specific features
    rag_config = generate_rag_monitoring_config()
    rag_metrics = rag_config.get('metrics', [])
    
    print("RAG Metrics Available:")
    for metric in rag_metrics:
        print(f"  • {metric}")
    
    print(f"\nRAG Layout Features:")
    layout = rag_config.get('layout_grid', {})
    if layout.get('type') == 'sections':
        sections = layout.get('sections', [])
        for section in sections:
            print(f"  • {section['title']}: {len(section['widgets'])} widgets")


def demo_configuration_persistence():
    """Demonstrate configuration save/load."""
    print("\n5. 💾 CONFIGURATION PERSISTENCE")
    print("-" * 32)
    
    manager = get_template_manager()
    
    # Create a custom configuration
    custom_config = {
        "context": "monitoring",
        "metrics": ["Request Volume", "Latency"],
        "layout": "custom_grid",
        "theme": "dark",
        "refresh_interval": 2000
    }
    
    # Save configuration
    config_name = manager.save_configuration(
        UIContext.MONITORING, 
        custom_config, 
        "demo_config"
    )
    
    print(f"✅ Saved configuration: {config_name}")
    
    # List configurations
    configs = manager.list_configurations()
    print(f"📁 Available configurations: {len(configs)}")
    for config in configs:
        print(f"  • {config}")


def demo_extensibility_examples():
    """Show how to extend the system."""
    print("\n6. 🚀 EXTENSIBILITY EXAMPLES")
    print("-" * 30)
    
    print("Adding New Widget Type:")
    print("""
    # In _determine_widget_type():
    elif category == MetricCategory.CUSTOM:
        return "my_custom_widget"
    
    # In _determine_widget_size():
    "my_custom_widget": {"width": 3, "height": 2}
    
    # In _generate_chart_config():
    elif widget_type == "my_custom_widget":
        base_config.update({"custom_option": True})
    """)
    
    print("Adding New UI Context:")
    print("""
    # In UIContext enum:
    NEW_ANALYSIS = "new_analysis"
    
    # In _load_default_configurations():
    UIContext.NEW_ANALYSIS: {
        "metrics": [...],
        "layout": "custom_layout",
        "refresh_interval": 5000
    }
    """)
    
    print("Adding New Layout Type:")
    print("""
    # In _generate_layout_grid():
    elif layout_type == "my_custom_layout":
        return {
            "type": "my_custom_layout",
            "config": {...}
        }
    """)


def main():
    """Run the complete features demo."""
    try:
        demo_extensible_template_system()
        demo_context_aware_configuration()
        demo_widget_generation()
        demo_rag_integration()
        demo_configuration_persistence()
        demo_extensibility_examples()
        
        print("\n" + "=" * 50)
        print("✨ MERIT Dashboard Features Summary:")
        print("✅ Extensible template management")
        print("✅ Context-aware metric configuration")
        print("✅ Automatic widget generation")
        print("✅ RAG metrics integration")
        print("✅ Configuration persistence")
        print("✅ Easy extensibility patterns")
        print("✅ Real-time monitoring support")
        print("✅ Responsive design system")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
