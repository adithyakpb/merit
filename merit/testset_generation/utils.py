"""
Utility functions for test set generation.

This module provides utility functions for validation, similarity calculation,
and parallel processing.
"""

import os
import threading
import concurrent.futures
from typing import Any, List, Callable, Dict
from functools import partial
import logging

from ..core.logging import get_logger
from ..core.models import Document

logger = get_logger(__name__)

# Thread-local storage for client instances
_thread_local = threading.local()

def get_thread_safe_client(original_client):
    """
    Get a thread-safe client instance.
    
    Args:
        original_client: The original client to make thread-safe
        
    Returns:
        A thread-safe client instance
    """
    if not hasattr(_thread_local, 'client'):
        # Clone the client for this thread
        # Note: This assumes the client can be cloned or is thread-safe
        _thread_local.client = original_client
    return _thread_local.client

def parallel_map(func: Callable, items: List, max_workers: int = None, 
                 use_processes: bool = False, **kwargs) -> List:
    """
    Execute a function on a list of items in parallel.
    
    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum number of workers (default: CPU count * 2)
        use_processes: Whether to use processes instead of threads
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        List of results
    """
    if not items:
        return []
    
    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 2 or 4)
    
    # Create a partial function with the kwargs
    if kwargs:
        func = partial(func, **kwargs)
    
    # Choose the executor based on the task type
    executor_class = concurrent.futures.ProcessPoolExecutor if use_processes else concurrent.futures.ThreadPoolExecutor
    
    results = []
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(func, item): i for i, item in enumerate(items)}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")
                results.append((idx, None))
    
    # Sort results by original index
    results.sort(key=lambda x: x[0])
    return [r for _, r in results]

def is_valid_input(text: str) -> bool:
    """
    Check if a string is a valid input.
    
    Args:
        text: The string to check.
        
    Returns:
        bool: True if the string is a valid input, False otherwise.
    """
    # Check if text is a string
    if not isinstance(text, str):
        return False
    
    # Minimum length check
    if len(text) < 10:
        return False
    
    # Check for JSON markers
    if text.startswith(("```", "[", "{")) or text in ["[", "{", "```", "```json"]:
        return False
    
    # Check that it contains actual words (at least 3)
    words = [w for w in text.split() if len(w) > 1]
    if len(words) < 3:
        return False
    
    return True


def find_optimal_clusters(features, max_clusters: int = 20, random_state: int = 42) -> int:
    import numpy as np
    """
    Find the optimal number of clusters using the elbow method.
    
    Args:
        features: Feature matrix
        max_clusters: Maximum number of clusters to consider
        random_state: Random seed for reproducibility
        
    Returns:
        int: Optimal number of clusters
    """
    
    # Limit max clusters based on data size
    max_clusters = min(max_clusters, len(features) // 5)
    max_clusters = max(2, min(10, max_clusters))  # At least 2, at most 10
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        # Calculate inertias and silhouette scores for different cluster counts
        inertias = []
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(features, labels))
        # Try to use kneed package for elbow detection
        try:
            from kneed import KneeLocator
            kneedle = KneeLocator( range(2, max_clusters + 1), inertias, S=1.0, curve="convex", direction="decreasing" )
            elbow = kneedle.elbow
            
            # If no clear elbow, use max silhouette score
            if elbow is None:
                return np.argmax(silhouette_scores) + 2
            return elbow
        except ImportError:
            # If kneed is not available, use silhouette score
            return np.argmax(silhouette_scores) + 2
    
    except ImportError:
        # If sklearn is not available, use a simple heuristic
        return min(5, len(features) // 20 + 2)

def compute_quality_metrics(
    original_features,
    selected_indices: List[int],
    clustering_results: Dict[str, Any],
    semantic_embeddings
) -> Dict[str, Any]:
    import numpy as np
    """
    Compute quality metrics for the selected subset.
    
    Args:
        original_features: Original feature matrix
        selected_indices: Indices of selected examples
        clustering_results: Results from clustering
        semantic_embeddings: Semantic embeddings
        
    Returns:
        Dict[str, Any]: Quality metrics
    """
    import numpy as np
    
    # Extract selected features
    selected_features = original_features[selected_indices]
    
    # Basic metrics
    metrics = {
        "coverage_ratio": len(selected_indices) / len(original_features)
    }
    
    # Try to compute convex hull coverage if scipy is available
    try:
        from scipy.spatial import ConvexHull
        from sklearn.decomposition import PCA
        
        # Use PCA to reduce dimensionality for ConvexHull
        n_components = min(selected_features.shape[0] - 1, selected_features.shape[1], 10)
        if n_components >= 2:
            pca = PCA(n_components=n_components)
            original_reduced = pca.fit_transform(original_features)
            subset_reduced = pca.transform(selected_features)
            
            try:
                original_hull = ConvexHull(original_reduced)
                subset_hull = ConvexHull(subset_reduced)
                metrics["hull_coverage"] = subset_hull.volume / original_hull.volume
            except Exception:
                # Silently fail if ConvexHull fails
                pass
    except ImportError:
        # Skip if scipy is not available
        pass
    
    # Distribution similarity using Wasserstein distance if available
    try:
        from scipy.stats import wasserstein_distance
        distances = []
        for dim in range(original_features.shape[1]):
            orig_values = original_features[:, dim]
            subset_values = selected_features[:, dim]
            # Add small epsilon to avoid division by zero
            distances.append(1.0 / (1.0 + wasserstein_distance(orig_values, subset_values)))
        
        metrics["distribution_similarity"] = np.mean(distances)
    except ImportError:
        # Fallback to simpler metric
        orig_mean = np.mean(original_features, axis=0)
        subset_mean = np.mean(selected_features, axis=0)
        metrics["distribution_similarity"] = 1.0 / (1.0 + np.linalg.norm(orig_mean - subset_mean))
    
    # Cluster representation
    cluster_labels = clustering_results["labels"]
    unique_clusters = clustering_results["unique_clusters"]
    cluster_counts = clustering_results["counts"]
    
    subset_cluster_counts = {
        i: np.sum(cluster_labels[selected_indices] == i) 
        for i in unique_clusters
    }
    
    proportional_representation = {}
    for cluster_id in cluster_counts:
        original_prop = cluster_counts[cluster_id] / len(original_features)
        subset_prop = (
            subset_cluster_counts[cluster_id] / len(selected_indices) 
            if subset_cluster_counts[cluster_id] > 0 else 0
        )
        proportional_representation[cluster_id] = 1.0 - abs(original_prop - subset_prop)
    
    metrics["proportional_representation"] = proportional_representation
    metrics["average_proportional_representation"] = np.mean(list(proportional_representation.values()))
    
    # Semantic diversity
    selected_semantics = semantic_embeddings[selected_indices]
    
    # Calculate average pairwise distance as diversity measure
    if len(selected_semantics) > 1:
        total_distance = 0
        count = 0
        for i in range(len(selected_semantics)):
            for j in range(i + 1, len(selected_semantics)):
                total_distance += np.linalg.norm(selected_semantics[i] - selected_semantics[j])
                count += 1
        
        metrics["semantic_diversity"] = total_distance / count if count > 0 else 0
    else:
        metrics["semantic_diversity"] = 0
    
    return metrics

def print_diagnostics(
    metrics: Dict[str, Any],
    cluster_distribution: Dict[int, int],
    sample_distribution: Dict[int, int],
    silhouette_avg: float
) -> None:
    """
    Print diagnostic information about the selection process.
     Args:
        metrics: Quality metrics
        cluster_distribution: Distribution of examples across clusters
        sample_distribution: Distribution of selected examples across clusters
        silhouette_avg: Average silhouette score
    """
    print(f"Silhouette Score: {silhouette_avg:.4f}")

    print("\nCluster Distribution:")
    for cluster_id, count in sorted(cluster_distribution.items()):
        original_percent = count / sum(cluster_distribution.values()) * 100
        sample_count = sample_distribution.get(cluster_id, 0)
        sample_total = sum(sample_distribution.values())
        sample_percent = (sample_count / sample_total * 100) if sample_total > 0 else 0
        print(f"  Cluster {cluster_id}: {count} examples ({original_percent:.1f}%) → {sample_count} samples ({sample_percent:.1f}%)")

    print("\nQuality Metrics:")
    print(f"  Coverage Ratio: {metrics.get('coverage_ratio', 'N/A'):.4f}")
    print(f"  Hull Coverage: {metrics.get('hull_coverage', 'N/A')}")
    print(f"  Distribution Similarity: {metrics.get('distribution_similarity', 'N/A'):.4f}")
    print(f"  Average Proportional Representation: {metrics.get('average_proportional_representation', 'N/A'):.4f}")
    print(f"  Semantic Diversity: {metrics.get('semantic_diversity', 'N/A'):.4f}")