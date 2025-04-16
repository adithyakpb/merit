"""
Monitoring-specific metric implementations for MERIT.

This module provides base classes and implementations for metrics
that are primarily used in the monitoring context.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from .base import (
    BaseMetric, 
    MetricContext, 
    MetricCategory,
    register_metric
)


class MonitoringMetric(BaseMetric):
    """
    Base class for monitoring-specific metrics.
    
    These metrics are designed to work with monitoring data and focus on
    operational aspects rather than evaluation quality.
    """
    context = MetricContext.MONITORING
    
    def calculate_time_series(self, 
                              data: List[Dict[str, Any]], 
                              time_window: Optional[timedelta] = None,
                              interval: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Calculate a time series of this metric over a dataset.
        
        Args:
            data: List of interaction data points
            time_window: Optional time window to restrict data
            interval: Optional interval for bucketing (e.g., "1h", "1d")
            
        Returns:
            List of metric results over time
        """
        # This is a default implementation that calculates the metric 
        # for each time bucket. Subclasses may override for efficiency.
        
        # Group data by time buckets
        buckets = self._group_by_time(data, interval)
        
        # Calculate metric for each bucket
        results = []
        for timestamp, bucket_data in buckets.items():
            result = self(bucket_data)
            results.append({
                "timestamp": timestamp,
                "value": result.get("value"),
                "count": len(bucket_data),
                **{k: v for k, v in result.items() if k != "value"}
            })
            
        return sorted(results, key=lambda x: x["timestamp"])
    
    def _group_by_time(self, 
                      data: List[Dict[str, Any]], 
                      interval: Optional[str] = None) -> Dict[datetime, List[Dict[str, Any]]]:
        """
        Group data by time buckets.
        
        Args:
            data: List of interaction data points
            interval: Interval for bucketing (e.g., "1h", "1d")
            
        Returns:
            Dictionary mapping timestamps to lists of data points
        """
        if not interval:
            interval = "1h"  # Default to hourly
            
        # Parse interval string
        if interval.endswith("m"):
            delta = timedelta(minutes=int(interval[:-1]))
        elif interval.endswith("h"):
            delta = timedelta(hours=int(interval[:-1]))
        elif interval.endswith("d"):
            delta = timedelta(days=int(interval[:-1]))
        else:
            raise ValueError(f"Invalid interval format: {interval}")
        
        # Group data by time buckets
        buckets = {}
        for item in data:
            # Get timestamp from data
            if "timestamp" in item:
                try:
                    if isinstance(item["timestamp"], str):
                        timestamp = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                    else:
                        timestamp = item["timestamp"]
                except (ValueError, TypeError):
                    # Skip items with invalid timestamps
                    continue
            else:
                # Skip items without timestamps
                continue
                
            # Calculate bucket timestamp (start of interval)
            if isinstance(delta, timedelta):
                # For hour and minute intervals
                bucket_time = timestamp.replace(
                    minute=0 if delta >= timedelta(hours=1) else (timestamp.minute // int(delta.total_seconds() // 60) * int(delta.total_seconds() // 60)),
                    second=0,
                    microsecond=0
                )
            else:
                # For day intervals
                bucket_time = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                
            # Add to bucket
            if bucket_time not in buckets:
                buckets[bucket_time] = []
            buckets[bucket_time].append(item)
            
        return buckets


class PerformanceMetric(MonitoringMetric):
    """Base class for performance-related monitoring metrics."""
    category = MetricCategory.PERFORMANCE


class UsageMetric(MonitoringMetric):
    """Base class for usage-related monitoring metrics."""
    category = MetricCategory.USAGE


class CostMetric(MonitoringMetric):
    """Base class for cost-related monitoring metrics."""
    category = MetricCategory.COST


class RequestVolumeMetric(UsageMetric):
    """
    Measures the volume of requests over time.
    
    This metric is useful for understanding traffic patterns and capacity planning.
    """
    name = "Request Volume"
    description = "Number of requests over a time period"
    greater_is_better = None  # Neutral - depends on capacity planning
    
    def __call__(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate the request volume.
        
        Args:
            interactions: List of interaction data
            
        Returns:
            Dict with volume metrics
        """
        # Simply count the number of interactions
        count = len(interactions)
        
        # Get time range if available
        start_time = None
        end_time = None
        
        for interaction in interactions:
            timestamp = interaction.get("timestamp")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    else:
                        dt = timestamp
                        
                    if start_time is None or dt < start_time:
                        start_time = dt
                    if end_time is None or dt > end_time:
                        end_time = dt
                except (ValueError, TypeError):
                    pass
        
        result = {
            "value": count
        }
        
        # Add time range if available
        if start_time and end_time:
            result["start_time"] = start_time.isoformat()
            result["end_time"] = end_time.isoformat()
            
            # Calculate rate if time span > 0
            time_diff = (end_time - start_time).total_seconds()
            if time_diff > 0:
                result["rate"] = count / time_diff  # Requests per second
                
        return result


class LatencyMetric(PerformanceMetric):
    """
    Measures the latency of LLM responses.
    
    This metric is useful for understanding performance and user experience.
    """
    name = "Latency"
    description = "Response time in seconds"
    greater_is_better = False  # Lower latency is better
    
    def __call__(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate latency statistics.
        
        Args:
            interactions: List of interaction data
            
        Returns:
            Dict with latency metrics
        """
        # Extract latency values
        latencies = []
        for interaction in interactions:
            # Try different ways to get latency
            latency = None
            
            # Direct latency field
            if "latency" in interaction:
                latency = interaction["latency"]
            
            # Response object
            elif "response" in interaction and isinstance(interaction["response"], dict):
                latency = interaction["response"].get("latency")
            
            # Calculate from timestamps
            elif all(k in interaction for k in ["request_timestamp", "response_timestamp"]):
                try:
                    req_time = datetime.fromisoformat(interaction["request_timestamp"].replace("Z", "+00:00"))
                    resp_time = datetime.fromisoformat(interaction["response_timestamp"].replace("Z", "+00:00"))
                    latency = (resp_time - req_time).total_seconds()
                except (ValueError, TypeError):
                    pass
            
            if latency is not None and isinstance(latency, (int, float)) and latency >= 0:
                latencies.append(latency)
        
        # Calculate statistics
        if not latencies:
            return {"value": None, "count": 0}
        
        avg_latency = sum(latencies) / len(latencies)
        latencies.sort()
        
        median_latency = latencies[len(latencies) // 2]
        if len(latencies) % 2 == 0 and len(latencies) > 1:
            median_latency = (median_latency + latencies[len(latencies) // 2 - 1]) / 2
            
        p95_idx = int(len(latencies) * 0.95)
        p95_latency = latencies[min(p95_idx, len(latencies) - 1)]
        
        # Construct result
        return {
            "value": avg_latency,
            "count": len(latencies),
            "min": latencies[0],
            "max": latencies[-1],
            "median": median_latency,
            "p95": p95_latency
        }


class TokenVolumeMetric(UsageMetric):
    """
    Measures the volume of tokens used in LLM interactions.
    
    This metric is useful for understanding usage patterns and costs.
    """
    name = "Token Volume"
    description = "Number of tokens used in LLM interactions"
    greater_is_better = None  # Neutral - depends on business goals
    
    def __call__(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate token usage statistics.
        
        Args:
            interactions: List of interaction data
            
        Returns:
            Dict with token usage metrics
        """
        # Initialize counters
        total_input_tokens = 0
        total_output_tokens = 0
        total_interactions = 0
        interactions_with_tokens = 0
        
        # Process interactions
        for interaction in interactions:
            total_interactions += 1
            
            # Try different ways to get token info
            input_tokens = None
            output_tokens = None
            
            # Direct token fields
            if "input_tokens" in interaction and "output_tokens" in interaction:
                input_tokens = interaction["input_tokens"]
                output_tokens = interaction["output_tokens"]
            
            # Tokens object
            elif "tokens" in interaction and isinstance(interaction["tokens"], dict):
                input_tokens = interaction["tokens"].get("input_tokens") or interaction["tokens"].get("prompt_tokens")
                output_tokens = interaction["tokens"].get("output_tokens") or interaction["tokens"].get("completion_tokens")
            
            # Response object
            elif "response" in interaction and isinstance(interaction["response"], dict):
                if "tokens" in interaction["response"] and isinstance(interaction["response"]["tokens"], dict):
                    input_tokens = interaction["response"]["tokens"].get("input_tokens") or interaction["response"]["tokens"].get("prompt_tokens")
                    output_tokens = interaction["response"]["tokens"].get("output_tokens") or interaction["response"]["tokens"].get("completion_tokens")
            
            # Add to totals if we found token info
            if input_tokens is not None and output_tokens is not None:
                try:
                    input_tokens = int(input_tokens)
                    output_tokens = int(output_tokens)
                    
                    if input_tokens >= 0 and output_tokens >= 0:
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        interactions_with_tokens += 1
                except (ValueError, TypeError):
                    pass
        
        # Construct result
        result = {
            "value": total_input_tokens + total_output_tokens,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_interactions": total_interactions,
            "interactions_with_tokens": interactions_with_tokens
        }
        
        # Calculate averages if we have data
        if interactions_with_tokens > 0:
            result["avg_input_tokens"] = total_input_tokens / interactions_with_tokens
            result["avg_output_tokens"] = total_output_tokens / interactions_with_tokens
            result["avg_total_tokens"] = (total_input_tokens + total_output_tokens) / interactions_with_tokens
            
        return result


class ErrorRateMetric(PerformanceMetric):
    """
    Measures the rate of errors in LLM interactions.
    
    This metric is useful for understanding reliability and troubleshooting.
    """
    name = "Error Rate"
    description = "Percentage of requests that resulted in errors"
    greater_is_better = False  # Lower error rate is better
    
    def __call__(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate error rate.
        
        Args:
            interactions: List of interaction data
            
        Returns:
            Dict with error rate metrics
        """
        if not interactions:
            return {"value": None, "count": 0}
            
        total_interactions = len(interactions)
        error_count = 0
        error_types = {}
        
        # Process interactions
        for interaction in interactions:
            # Try different ways to detect errors
            
            # Status field
            if "status" in interaction:
                if interaction["status"] not in ["success", "successful", 200, "200"]:
                    error_count += 1
                    error_type = str(interaction["status"])
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Response object
            elif "response" in interaction and isinstance(interaction["response"], dict):
                if interaction["response"].get("status") not in ["success", "successful", 200, "200"]:
                    error_count += 1
                    error_type = str(interaction["response"].get("status"))
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Error field
            elif "error" in interaction and interaction["error"]:
                error_count += 1
                error_type = str(interaction["error"])
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Calculate error rate
        error_rate = error_count / total_interactions if total_interactions > 0 else 0
        
        # Construct result
        return {
            "value": error_rate,
            "count": total_interactions,
            "error_count": error_count,
            "error_types": error_types
        }


class CostEstimateMetric(CostMetric):
    """
    Estimates the cost of LLM usage based on token counts and model pricing.
    
    This metric is useful for budget tracking and cost optimization.
    """
    name = "Cost Estimate"
    description = "Estimated cost of LLM usage in USD"
    greater_is_better = False  # Lower cost is better
    
    def __init__(self, pricing_config: Optional[Dict[str, Any]] = None):
        """
        Initialize with pricing configuration.
        
        Args:
            pricing_config: Optional dictionary mapping model names to pricing info
        """
        self.pricing_config = pricing_config or self._get_default_pricing()
    
    def __call__(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate cost estimate.
        
        Args:
            interactions: List of interaction data
            
        Returns:
            Dict with cost metrics
        """
        if not interactions:
            return {"value": 0.0, "count": 0}
            
        total_cost = 0.0
        costs_by_model = {}
        interactions_with_cost = 0
        
        # Process interactions
        for interaction in interactions:
            # Get model name
            model = None
            if "model" in interaction:
                model = interaction["model"]
            elif "request" in interaction and isinstance(interaction["request"], dict):
                model = interaction["request"].get("model")
            elif "response" in interaction and isinstance(interaction["response"], dict):
                model = interaction["response"].get("model")
            
            # Skip if no model info
            if not model:
                continue
                
            # Get token counts
            input_tokens = None
            output_tokens = None
            
            # Direct token fields
            if "input_tokens" in interaction and "output_tokens" in interaction:
                input_tokens = interaction["input_tokens"]
                output_tokens = interaction["output_tokens"]
            
            # Tokens object
            elif "tokens" in interaction and isinstance(interaction["tokens"], dict):
                input_tokens = interaction["tokens"].get("input_tokens") or interaction["tokens"].get("prompt_tokens")
                output_tokens = interaction["tokens"].get("output_tokens") or interaction["tokens"].get("completion_tokens")
            
            # Response object
            elif "response" in interaction and isinstance(interaction["response"], dict):
                if "tokens" in interaction["response"] and isinstance(interaction["response"]["tokens"], dict):
                    input_tokens = interaction["response"]["tokens"].get("input_tokens") or interaction["response"]["tokens"].get("prompt_tokens")
                    output_tokens = interaction["response"]["tokens"].get("output_tokens") or interaction["response"]["tokens"].get("completion_tokens")
            
            # Skip if no token info
            if input_tokens is None or output_tokens is None:
                continue
                
            try:
                input_tokens = int(input_tokens)
                output_tokens = int(output_tokens)
                
                if input_tokens < 0 or output_tokens < 0:
                    continue
            except (ValueError, TypeError):
                continue
            
            # Get pricing for this model
            pricing = self._get_model_pricing(model)
            if not pricing:
                continue
                
            # Calculate cost
            input_cost = input_tokens * pricing["input_price_per_token"]
            output_cost = output_tokens * pricing["output_price_per_token"]
            interaction_cost = input_cost + output_cost
            
            # Update totals
            total_cost += interaction_cost
            interactions_with_cost += 1
            
            # Update costs by model
            if model not in costs_by_model:
                costs_by_model[model] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "count": 0
                }
            
            costs_by_model[model]["cost"] += interaction_cost
            costs_by_model[model]["input_tokens"] += input_tokens
            costs_by_model[model]["output_tokens"] += output_tokens
            costs_by_model[model]["count"] += 1
        
        # Construct result
        return {
            "value": total_cost,
            "count": interactions_with_cost,
            "currency": "USD",
            "costs_by_model": costs_by_model
        }
    
    def _get_model_pricing(self, model: str) -> Optional[Dict[str, float]]:
        """
        Get pricing information for a model.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with pricing info or None if not available
        """
        # Try exact match
        if model in self.pricing_config:
            return self.pricing_config[model]
            
        # Try prefix match
        model_lower = model.lower()
        for prefix, pricing in self.pricing_config.items():
            if model_lower.startswith(prefix.lower()):
                return pricing
                
        # No match
        return None
    
    def _get_default_pricing(self) -> Dict[str, Dict[str, float]]:
        """
        Get default pricing information for common models.
        
        Returns:
            Dictionary mapping model names to pricing info
        """
        return {
            # OpenAI models
            "gpt-4": {
                "input_price_per_token": 0.00003,
                "output_price_per_token": 0.00006
            },
            "gpt-4-32k": {
                "input_price_per_token": 0.00006,
                "output_price_per_token": 0.00012
            },
            "gpt-3.5-turbo": {
                "input_price_per_token": 0.0000015,
                "output_price_per_token": 0.000002
            },
            # Anthropic models
            "claude-2": {
                "input_price_per_token": 0.00001102,
                "output_price_per_token": 0.00003268
            },
            "claude-instant-1": {
                "input_price_per_token": 0.00000163,
                "output_price_per_token": 0.00000551
            }
            # Add more models as needed
        }


# Register the metrics
register_metric(RequestVolumeMetric)
register_metric(LatencyMetric)
register_metric(TokenVolumeMetric)
register_metric(ErrorRateMetric)
register_metric(CostEstimateMetric)
