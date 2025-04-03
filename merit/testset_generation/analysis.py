"""
Example analysis functionality.

This module provides functions for analyzing example inputs to extract
patterns, styles, and other characteristics.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Union

from ..core.cache import cache_analysis, is_caching_available
from ..core.logging import get_logger
from ..core.utils import parse_json
from ..core.models import ExampleInputSet
from ..core.prompts import INPUT_STYLE_ANALYSIS_PROMPT

logger = get_logger(__name__)

@cache_analysis
def analyze_examples(
    inputs: List[str],
    client: Any = None,
    use_llm: bool = False,
    max_inputs_for_llm: int = 20,
    analysis_type: str = "all"
) -> Dict[str, Any]:
    """
    Comprehensive analysis of example inputs (inputs, commands, statements, etc.).

    This function combines the functionality of analyze_input_patterns and
    analyze_example_inputs into a single unified interface.

    Args:
        inputs: List of input strings or ExampleInputSet to analyze
        client: The GenericAPIClient for text generation (required for LLM-based analysis)
        use_llm: Whether to use LLM for advanced style analysis
        max_inputs_for_llm: Maximum number of inputs to send to LLM
        analysis_type: Type of analysis to perform:
            - "basic": Only basic metrics (length, input types)
            - "comprehensive": Detailed analysis without LLM
            - "all": Full analysis including LLM-based style analysis (default)

    Returns:
        Dict[str, Any]: Analysis results
    """
    logger.info(f"Analyzing {len(inputs)} examples with analysis_type={analysis_type}")

    # Initialize analysis with all required keys to prevent KeyError
    analysis = {
        "input_types": {
            "question": 0,
            "statement": 0, 
            "command": 0,
            "other": 0
        },
        "question_types": {
            "what": 0,
            "how": 0,
            "why": 0,
            "when": 0,
            "where": 0,
            "who": 0,
            "which": 0,
            "yes_no": 0,
            "other": 0
        },
        "common_phrases": {},
        "avg_length": 0,
        "average_length": 0  # For backward compatibility
    }

    try:
        # Basic metrics (always calculated)
        total_length = 0
        total_words = 0
        word_frequencies = {}

        # Common command verbs for detection
        command_verbs = ["find", "show", "tell", "list", "search", "get", "describe", "explain", "summarize"]

        # Analyze n-grams if doing comprehensive analysis
        bigrams = {}

        # First word analysis
        first_words = {}

        for input_text in inputs:
            # Length analysis
            total_length += len(input_text)
            words = input_text.split()
            total_words += len(words)

            # First word analysis
            first_word = words[0].lower() if words else ""
            first_words[first_word] = first_words.get(first_word, 0) + 1

            # Input type analysis
            if input_text.strip().endswith('?'):
                analysis["input_types"]["question"] += 1
            elif first_word.lower() in command_verbs:
                analysis["input_types"]["command"] += 1
            elif input_text.strip().endswith('.'):
                analysis["input_types"]["statement"] += 1
            else:
                analysis["input_types"]["other"] += 1

            # For comprehensive or full analysis, do more detailed analysis
            if analysis_type in ["comprehensive", "all"]:
                # Question type analysis
                if input_text.strip().endswith('?'):
                    if first_word.lower() == "what":
                        analysis["question_types"]["what"] += 1
                    elif first_word.lower() == "how":
                        analysis["question_types"]["how"] += 1
                    elif first_word.lower() == "why":
                        analysis["question_types"]["why"] += 1
                    elif first_word.lower() == "when":
                        analysis["question_types"]["when"] += 1
                    elif first_word.lower() == "where":
                        analysis["question_types"]["where"] += 1
                    elif first_word.lower() == "who":
                        analysis["question_types"]["who"] += 1
                    elif first_word.lower() == "which":
                        analysis["question_types"]["which"] += 1
                    elif first_word.lower() in ["is", "are", "do", "does", "can", "could", "will", "would"]:
                        analysis["question_types"]["yes_no"] += 1
                    else:
                        analysis["question_types"]["other"] += 1

                # Word frequency analysis
                for word in words:
                    word = word.lower()
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1

                # Bigram analysis
                if len(words) > 1:
                    for i in range(len(words) - 1):
                        bigram = (words[i].lower(), words[i+1].lower())
                        bigrams[bigram] = bigrams.get(bigram, 0) + 1

        # Calculate averages
        if inputs:
            analysis["avg_length"] = total_length / len(inputs)
            analysis["average_length"] = analysis["avg_length"]  # For backward compatibility
            analysis["avg_word_count"] = total_words / len(inputs)

        # Add word frequencies
        analysis["word_frequencies"] = word_frequencies

        # Add first word distribution
        analysis["first_words"] = first_words

        # For comprehensive or full analysis, add more detailed analysis
        if analysis_type in ["comprehensive", "all"]:
            # Add common bigrams
            sorted_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
            analysis["common_bigrams"] = sorted_bigrams[:10]

            # Determine vocabulary richness
            unique_words = set(word_frequencies.keys())
            analysis["vocabulary_size"] = len(unique_words)
            analysis["vocabulary_richness"] = len(unique_words) / total_words if total_words else 0

        # Advanced LLM-based analysis (only if requested and client provided)
        if analysis_type == "all" and use_llm and client:
            # For large sets, select representative examples
            if len(inputs) > max_inputs_for_llm:
                selected_inputs = select_representative_examples(inputs, max_inputs_for_llm)
                logger.info(f"Selected {len(selected_inputs)} representative inputs for LLM analysis")
            else:
                selected_inputs = inputs

            try:
                # Format inputs for the prompt
                examples_str = "\n".join([f"- {inp}" for inp in selected_inputs])

                # Use the input style analysis prompt
                prompt = INPUT_STYLE_ANALYSIS_PROMPT.safe_format(
                    example_inputs=examples_str
                )

                # Get analysis from LLM
                response = client.generate_text(prompt)

                # Parse JSON response
                llm_analysis = parse_json(response, return_type="object")

                # Add LLM-based analysis
                analysis.update({
                    "structure_patterns": llm_analysis.get("structure_patterns", []),
                    "grammar_patterns": llm_analysis.get("grammar_patterns", []),
                    "vocabulary_level": llm_analysis.get("vocabulary_level", "unknown"),
                    "formality_level": llm_analysis.get("formality_level", 0.5),
                    "distinctive_elements": llm_analysis.get("distinctive_elements", []),
                    "example_templates": llm_analysis.get("example_templates", []),
                    "linguistic_features": llm_analysis.get("linguistic_features", {}),
                    "common_phrases": llm_analysis.get("common_phrases", []),
                })

                logger.info("Completed advanced style analysis of examples")
            except Exception as e:
                logger.error(f"Failed to perform LLM-based style analysis: {str(e)}")
                # Continue with basic analysis only

        logger.info(f"Completed analysis of {len(inputs)} examples")
        return analysis

    except Exception as e:
        logger.error(f"Failed to analyze examples: {str(e)}")
        return analysis

def _create_sub_clusters(features: np.ndarray, clustering_results: Dict[str, Any]) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Create sub-clusters within each main cluster.
    
    Args:
        features: Feature matrix
        clustering_results: Results from clustering
        
    Returns:
        Dict[int, Dict[int, np.ndarray]]: Mapping of main cluster IDs to sub-clusters
    """
    import numpy as np
    
    cluster_labels = clustering_results["labels"]
    sub_clusters = {}
    
    for cluster_id in clustering_results["unique_clusters"]:
        cluster_mask = (cluster_labels == cluster_id)
        cluster_features = features[cluster_mask]
        
        # Skip small clusters
        if len(cluster_features) < 5:
            sub_clusters[cluster_id] = {0: np.where(cluster_mask)[0]}
            continue
        
        # Determine appropriate number of sub-clusters
        n_sub_clusters = max(2, min(len(cluster_features) // 10, 5))
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            sub_clustering = AgglomerativeClustering(n_clusters=n_sub_clusters)
            sub_labels = sub_clustering.fit_predict(cluster_features)
            
            # Map back to original indices
            original_indices = np.where(cluster_mask)[0]
            sub_clusters[cluster_id] = {
                i: original_indices[sub_labels == i] 
                for i in range(n_sub_clusters)
            }
        except Exception:
            # Fallback if clustering fails
            sub_clusters[cluster_id] = {0: np.where(cluster_mask)[0]}
    
    return sub_clusters

def select_representative_examples(
    inputs: Union[List[str], 'ExampleInputSet'], 
    max_examples: int = 5,
    client: Any = None,
    method: str = "hybrid",
    semantic_weight: float = 0.7,
    structural_weight: float = 0.3,
    clustering_method: str = "kmeans",
    n_clusters: Union[int, str] = "auto",
    sampling_strategy: str = "stratified",
    include_edge_cases: bool = True,
    use_sub_clusters: bool = True,
    compute_metrics: bool = True,
    verbose: bool = False,
    return_metadata: bool = False
) -> Union[List[str], Tuple[List[str], Dict[str, Any]]]:
    """
    Select a diverse, representative subset of examples using multiple strategies.
    
    Args:
        inputs: List of input strings or ExampleSet
        max_examples: Maximum number of examples to select
        client: API client for embeddings (required for semantic clustering)
        method: Selection method:
            - "heuristic": Use only text-based heuristics (length, starting words, etc.)
            - "semantic": Use only semantic clustering based on embeddings
            - "hybrid": Combine both approaches (default)
        semantic_weight: Weight given to semantic features (0.0-1.0)
        structural_weight: Weight given to structural features (0.0-1.0)
        clustering_method: Clustering algorithm ("kmeans" or "hierarchical")
        sampling_strategy: Strategy for selecting examples ("stratified", "diversity", "coverage")
        include_edge_cases: Whether to include examples from cluster boundaries
        verbose: Whether to print progress information
        
    Returns:
        List[str]: Selected representative examples
    """
    import numpy as np
    
    # Process inputs if it's an ExampleSet
    if hasattr(inputs, 'inputs') and hasattr(inputs, 'to_dict'):
        # It's an ExampleSet
        input_objects = inputs.inputs
        inputs = [q.input for q in input_objects]
    
    if len(inputs) <= max_examples:
        return inputs
    
    # Set up logging based on verbose flag
    log_func = logger.info if verbose else lambda x: None
    
    # SEMANTIC CLUSTERING APPROACH
    if method in ["semantic", "hybrid"] and client is not None:
        try:
            log_func("Extracting features for semantic clustering...")
            
            # Get embeddings for all inputs
            embeddings = client.get_embeddings(inputs)
            
            # Extract structural features
            structural_features = []
            for inp in inputs:
                features = {
                    'length': len(inp),
                    'word_count': len(inp.split()),
                    'is_wh_input': int(any(inp.lower().startswith(wh) for wh in ['who', 'what', 'when', 'where', 'why', 'how'])),
                    'is_yes_no': int(inp.lower().startswith(('is', 'are', 'do', 'does', 'can', 'could', 'will', 'would'))),
                    'input_marks': inp.count('?'),
                    'has_multiple_inputs': int(inp.count('?') > 1),
                    'sentence_count': len([s for s in inp.split('.') if s]),
                    'capital_letters': sum(1 for c in inp if c.isupper()),
                    'contains_numbers': int(any(c.isdigit() for c in inp)),
                }
                structural_features.append(list(features.values()))
            
            # Convert to numpy array
            structural_features = np.array(structural_features)
            
            # Normalize features
            try:
                from sklearn.preprocessing import StandardScaler, normalize
                structural_features = StandardScaler().fit_transform(structural_features)
                semantic_embeddings_norm = normalize(embeddings)
                structural_features_norm = normalize(structural_features)
            except ImportError:
                # Fallback to simple normalization if sklearn is not available
                semantic_embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                # Simple z-score normalization for structural features
                structural_features_norm = (structural_features - np.mean(structural_features, axis=0)) / np.std(structural_features, axis=0)
                # Replace NaNs with 0
                structural_features_norm = np.nan_to_num(structural_features_norm)
            
            # Combine features with weights
            combined_features = np.concatenate([
                semantic_weight * semantic_embeddings_norm,
                structural_weight * structural_features_norm
            ], axis=1)
            
            # Determine number of clusters
            if n_clusters == "auto":
                log_func("Determining optimal number of clusters...")
                n_clusters = _find_optimal_clusters(combined_features)
                log_func(f"Optimal number of clusters determined: {n_clusters}")
            else:
                # Ensure n_clusters is an integer and within reasonable bounds
                n_clusters = int(n_clusters)
                n_clusters = max(2, min(n_clusters, len(inputs) // 2))
            
            log_func(f"Performing clustering with {n_clusters} clusters using {clustering_method}...")
            
            # Perform clustering
            try:
                if clustering_method == "hierarchical":
                    from sklearn.cluster import AgglomerativeClustering
                    clustering = AgglomerativeClustering(n_clusters=n_clusters)
                    cluster_labels = clustering.fit_predict(combined_features)
                elif clustering_method == "dbscan":
                    from sklearn.cluster import DBSCAN
                    from sklearn.neighbors import NearestNeighbors
                    
                    # Estimate eps parameter from data
                    nn = NearestNeighbors(n_neighbors=min(5, len(combined_features)-1))
                    nn.fit(combined_features)
                    distances, _ = nn.kneighbors(combined_features)
                    eps = np.percentile(distances[:, -1], 90)  # Use 90th percentile
                    
                    # Run DBSCAN
                    clustering = DBSCAN(eps=eps, min_samples=5)
                    cluster_labels = clustering.fit_predict(combined_features)
                    
                    # Handle noise points (-1 labels)
                    if -1 in cluster_labels:
                        noise_mask = (cluster_labels == -1)
                        valid_clusters = np.unique(cluster_labels[~noise_mask])
                        
                        if len(valid_clusters) > 0:  # If we have valid clusters
                            for idx in np.where(noise_mask)[0]:
                                # Assign to nearest cluster
                                distances = []
                                for cluster_id in valid_clusters:
                                    cluster_points = combined_features[cluster_labels == cluster_id]
                                    min_dist = np.min(np.linalg.norm(combined_features[idx] - cluster_points, axis=1))
                                    distances.append(min_dist)
                                
                                cluster_labels[idx] = valid_clusters[np.argmin(distances)]
                        else:
                            # If all points are noise, fall back to KMeans
                            from sklearn.cluster import KMeans
                            clustering = KMeans(n_clusters=n_clusters, random_state=42)
                            cluster_labels = clustering.fit_predict(combined_features)
                else:  # Default to kmeans
                    from sklearn.cluster import KMeans
                    clustering = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = clustering.fit_predict(combined_features)
                
                # Calculate silhouette scores if available
                try:
                    from sklearn.metrics import silhouette_samples, silhouette_score
                    silhouette_avg = silhouette_score(combined_features, cluster_labels)
                    sample_silhouette_values = silhouette_samples(combined_features, cluster_labels)
                    log_func(f"Silhouette score: {silhouette_avg:.4f}")
                except ImportError:
                    # Fallback if sklearn metrics not available
                    sample_silhouette_values = np.zeros(len(inputs))
                    silhouette_avg = 0
            except ImportError:
                log_func("Sklearn clustering not available, falling back to heuristic method")
                return _heuristic_selection(inputs, max_examples)
            
            # Select samples based on strategy
            selected_indices = []
            
            # Calculate cluster distribution
            unique_clusters = np.unique(cluster_labels)
            cluster_counts = {c: np.sum(cluster_labels == c) for c in unique_clusters}
            
            # Allocate samples based on sampling strategy
            if sampling_strategy == "stratified":
                # Proportional allocation
                allocation = {}
                for cluster_id in unique_clusters:
                    proportion = cluster_counts[cluster_id] / len(inputs)
                    allocated = max(1, int(max_examples * proportion))
                    allocation[cluster_id] = allocated
            elif sampling_strategy == "diversity":
                # Equal allocation to maximize diversity
                base_allocation = max_examples // len(unique_clusters)
                allocation = {cluster_id: base_allocation for cluster_id in unique_clusters}
                # Allocate remaining samples to larger clusters
                remaining = max_examples - base_allocation * len(unique_clusters)
                for cluster_id in sorted(cluster_counts, key=cluster_counts.get, reverse=True)[:remaining]:
                    allocation[cluster_id] += 1
            else:  # Default to "coverage"
                # Allocate more to clusters with higher variance
                cluster_variances = {}
                for cluster_id in unique_clusters:
                    cluster_mask = (cluster_labels == cluster_id)
                    cluster_features = combined_features[cluster_mask]
                    # Use sum of variances along each dimension
                    cluster_variances[cluster_id] = np.sum(np.var(cluster_features, axis=0))
                
                # Normalize variances
                total_variance = sum(cluster_variances.values())
                allocation = {}
                for cluster_id, variance in cluster_variances.items():
                    proportion = variance / total_variance if total_variance > 0 else 1.0 / len(unique_clusters)
                    allocated = max(1, int(max_examples * proportion))
                    allocation[cluster_id] = allocated
            
            # Adjust allocations to match max_examples
            while sum(allocation.values()) != max_examples:
                if sum(allocation.values()) > max_examples:
                    # Remove from largest cluster
                    largest_cluster = max(allocation, key=allocation.get)
                    if allocation[largest_cluster] > 1:  # Ensure at least 1 per cluster
                        allocation[largest_cluster] -= 1
                else:
                    # Add to smallest cluster
                    smallest_cluster = min(allocation, key=allocation.get)
                    allocation[smallest_cluster] += 1
            
            log_func("Selecting representative examples from each cluster...")
            
            # Create clustering results dictionary for sub-clustering
            clustering_results = {
                "labels": cluster_labels,
                "unique_clusters": unique_clusters,
                "counts": cluster_counts
            }
            
            # Create sub-clusters if enabled
            if use_sub_clusters:
                log_func("Creating sub-clusters for more precise selection...")
                sub_clusters = _create_sub_clusters(combined_features, clustering_results)
            else:
                # Create simple sub-clusters (each main cluster has one sub-cluster)
                sub_clusters = {}
                for cluster_id in unique_clusters:
                    cluster_mask = (cluster_labels == cluster_id)
                    sub_clusters[cluster_id] = {0: np.where(cluster_mask)[0]}
            
            # Select samples from each cluster/sub-cluster
            for cluster_id, n_to_select in allocation.items():
                # Get sub-clusters for this cluster
                cluster_sub_clusters = sub_clusters[cluster_id]
                
                # Allocate samples to sub-clusters
                sub_allocation = {}
                total_in_cluster = sum(len(indices) for indices in cluster_sub_clusters.values())
                
                for sub_id, indices in cluster_sub_clusters.items():
                    # Allocate proportionally to sub-cluster size
                    proportion = len(indices) / total_in_cluster
                    sub_allocation[sub_id] = max(1, int(n_to_select * proportion))
                
                # Adjust to match cluster allocation
                while sum(sub_allocation.values()) != n_to_select:
                    if sum(sub_allocation.values()) > n_to_select:
                        # Remove from largest sub-cluster
                        largest_sub = max(sub_allocation, key=sub_allocation.get)
                        if sub_allocation[largest_sub] > 1:  # Ensure at least 1 per sub-cluster
                            sub_allocation[largest_sub] -= 1
                        else:
                            # If all at minimum, remove from sub-cluster with most examples
                            largest_sub = max(cluster_sub_clusters, key=lambda x: len(cluster_sub_clusters[x]))
                            sub_allocation[largest_sub] -= 1
                    else:
                        # Add to smallest sub-cluster
                        smallest_sub = min(sub_allocation, key=sub_allocation.get)
                        sub_allocation[smallest_sub] += 1
                
                # Select from each sub-cluster
                for sub_id, n_sub_to_select in sub_allocation.items():
                    sub_indices = cluster_sub_clusters[sub_id]
                    
                    # Skip if sub-cluster is empty (shouldn't happen but just in case)
                    if len(sub_indices) == 0:
                        continue
                    
                    # Select medoid (center) example
                    sub_features = combined_features[sub_indices]
                    centroid = np.mean(sub_features, axis=0)
                    distances = np.linalg.norm(sub_features - centroid, axis=1)
                    medoid_idx = sub_indices[np.argmin(distances)]
                    selected_indices.append(medoid_idx)
                    
                    # Select remaining examples
                    remaining = n_sub_to_select - 1  # Subtract 1 for the medoid
                    if remaining > 0 and len(sub_indices) > 1:
                        remaining_indices = [idx for idx in sub_indices if idx != medoid_idx]
                        
                        if include_edge_cases and remaining > 1 and len(remaining_indices) > 1:
                            # Find silhouette scores for remaining indices
                            remaining_silhouettes = sample_silhouette_values[remaining_indices]
                            # Include boundary examples (low silhouette score)
                            boundary_idx = remaining_indices[np.argmin(remaining_silhouettes)]
                            selected_indices.append(boundary_idx)
                            remaining -= 1
                            remaining_indices = [idx for idx in remaining_indices if idx != boundary_idx]
                        
                        # Select diverse examples from remaining
                        if remaining > 0 and remaining_indices:
                            # Use maxmin sampling for diversity
                            diverse_selections = []
                            candidates = np.array(remaining_indices)
                            candidate_features = combined_features[candidates]
                            
                            while len(diverse_selections) < min(remaining, len(candidates)):
                                if len(diverse_selections) == 0:
                                    # For first selection, pick furthest from medoid
                                    medoid_feature = combined_features[medoid_idx].reshape(1, -1)
                                    distances = np.linalg.norm(candidate_features - medoid_feature, axis=1)
                                    idx = np.argmax(distances)
                                else:
                                    # For subsequent selections, use maxmin distance
                                    selected_features = combined_features[[medoid_idx] + [candidates[i] for i in diverse_selections]]
                                    distances = []
                                    
                                    for i in range(len(candidates)):
                                        if i not in diverse_selections:
                                            # Calculate min distance to any selected point
                                            point_distances = np.linalg.norm(
                                                candidate_features[i].reshape(1, -1) - selected_features, 
                                                axis=1
                                            )
                                            distances.append(np.min(point_distances))
                                        else:
                                            distances.append(-1)  # Already selected
                                    
                                    idx = np.argmax(distances)
                                
                                diverse_selections.append(idx)
                            
                            selected_indices.extend([candidates[i] for i in diverse_selections])
            
            # Ensure no duplicates
            selected_indices = list(set(selected_indices))
            
            # If we somehow got too many, remove those with middling silhouette scores
            if len(selected_indices) > max_examples:
                # Sort by silhouette score (keep highest and lowest)
                sorted_by_silhouette = sorted(selected_indices, key=lambda idx: abs(0.5 - sample_silhouette_values[idx]))
                selected_indices = sorted_by_silhouette[:max_examples]
            
            # If we got too few, add additional diverse examples
            if len(selected_indices) < max_examples:
                remaining = max_examples - len(selected_indices)
                candidates = [i for i in range(len(inputs)) if i not in selected_indices]
                
                if candidates:
                    candidate_features = combined_features[candidates]
                    selected_features = combined_features[selected_indices]
                    
                    additional = []
                    while len(additional) < min(remaining, len(candidates)):
                        distances = []
                        for i in range(len(candidates)):
                            if i not in additional:
                                point_distances = np.linalg.norm(
                                    candidate_features[i].reshape(1, -1) - selected_features, 
                                    axis=1
                                )
                                distances.append(np.min(point_distances))
                            else:
                                distances.append(-1)
                        
                        idx = np.argmax(distances)
                        additional.append(idx)
                        # Update selected_features for next iteration
                        selected_features = np.vstack([selected_features, candidate_features[idx]])
                    
                    selected_indices.extend([candidates[i] for i in additional])
            
            # Compute quality metrics if enabled
            if compute_metrics:
                log_func("Computing quality metrics...")
                quality_metrics = _compute_quality_metrics(
                    combined_features,
                    selected_indices,
                    clustering_results,
                    embeddings
                )
                
                # Print diagnostics if verbose
                if verbose:
                    _print_diagnostics(
                        quality_metrics,
                        cluster_counts,
                        {c: sum(cluster_labels[selected_indices] == c) for c in unique_clusters},
                        silhouette_avg
                    )
            else:
                quality_metrics = {"computed": False}
            
            # Prepare final results
            selected_examples = [inputs[i] for i in selected_indices]
            
            log_func(f"Selected {len(selected_indices)} representative examples")
            
            # Return results with or without metadata
            if return_metadata:
                metadata = {
                    "input_info": {
                        "original_count": len(inputs),
                        "selected_count": len(selected_examples),
                        "input_type": "ExampleSet" if hasattr(inputs, 'inputs') else "list"
                    },
                    "features": {
                        "semantic_weight": semantic_weight,
                        "structural_weight": structural_weight,
                        "dimensions": {
                            "semantic": embeddings.shape[1],
                            "structural": structural_features.shape[1],
                            "combined": combined_features.shape[1]
                        }
                    },
                    "clustering": {
                        "method": clustering_method,
                        "n_clusters": len(unique_clusters),
                        "silhouette_score": silhouette_avg,
                        "cluster_distribution": cluster_counts,
                        "sample_distribution": {
                            c: sum(cluster_labels[selected_indices] == c) 
                            for c in unique_clusters
                        }
                    },
                    "selection": {
                        "strategy": sampling_strategy,
                        "indices": selected_indices,
                        "include_edge_cases": include_edge_cases,
                        "use_sub_clusters": use_sub_clusters
                    },
                    "metrics": quality_metrics
                }
                return selected_examples, metadata
            
            return selected_examples
            
        except Exception as e:
            # Log error and fall back to heuristic method
            logger.warning(f"Semantic clustering failed: {str(e)}. Falling back to heuristic method.")
            if verbose:
                import traceback
                logger.warning(traceback.format_exc())
    
    # If we're here, either:
    # 1. We're using only the heuristic method, or
    # 2. We're using hybrid/semantic method but it failed
    return _heuristic_selection(inputs, max_examples)

def _heuristic_selection(inputs: List[str], max_examples: int = 5) -> List[str]:
    """
    Select representative examples using heuristic methods (original implementation).
    
    Args:
        inputs: List of input strings
        max_examples: Maximum number of examples to select
        
    Returns:
        List[str]: Selected representative examples
    """
    if len(inputs) <= max_examples:
        return inputs
    
    # Strategy 1: Select examples with different lengths
    selected = []
    
    # Get short, medium, and long examples
    lengths = [(i, len(inp)) for i, inp in enumerate(inputs)]
    lengths.sort(key=lambda x: x[1])
    
    # Get one short, one medium, and one long example
    if lengths:
        short_idx = lengths[0][0]
        medium_idx = lengths[len(lengths)//2][0]
        long_idx = lengths[-1][0]
        selected.extend([inputs[short_idx], inputs[medium_idx], inputs[long_idx]])
    
    # Strategy 2: Add examples with different starting words if we have room
    start_words = {}
    for i, inp in enumerate(inputs):
        words = inp.split()
        if words:
            start_word = words[0].lower()
            if start_word not in start_words and len(selected) < max_examples:
                if i not in [short_idx, medium_idx, long_idx]:  # Avoid duplicates
                    start_words[start_word] = i
                    selected.append(inputs[i])
    
    # Strategy 3: Add examples with different ending punctuation if we have room
    if len(selected) < max_examples:
        has_input = any(inp.strip().endswith('?') for inp in selected)
        has_period = any(inp.strip().endswith('.') for inp in selected)
        has_exclamation = any(inp.strip().endswith('!') for inp in selected)
        
        for i, inp in enumerate(inputs):
            if inp not in selected and len(selected) < max_examples:
                if not has_input and inp.strip().endswith('?'):
                    selected.append(inp)
                    has_input = True
                elif not has_period and inp.strip().endswith('.'):
                    selected.append(inp)
                    has_period = True
                elif not has_exclamation and inp.strip().endswith('!'):
                    selected.append(inp)
                    has_exclamation = True
    
    # If we still have room, add random examples
    import random
    while len(selected) < max_examples and len(selected) < len(inputs):
        idx = random.randint(0, len(inputs) - 1)
        if inputs[idx] not in selected:
            selected.append(inputs[idx])
    
    return selected




def remove_similar_inputs(
    inputs: List[str],
    client: Any,
    similarity_threshold: float = 0.85,
) -> List[str]:
    """
    Remove similar inputs from a list of inputs.
    
    Args:
        inputs: The inputs to filter.
        client: The API client to use for embeddings.
        similarity_threshold: The threshold for similarity detection.
        
    Returns:
        List[str]: The filtered inputs.
    """
    if len(inputs) <= 1:
        return inputs
    
    logger.info(f"Removing similar inputs from {len(inputs)} inputs")
    
    try:
        # Get embeddings for all inputs
        embeddings = client.get_embeddings(inputs)
        
        # Calculate similarity matrix
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        # Pre-compute norms for efficiency
        norms = np.array([np.linalg.norm(emb) for emb in embeddings])
        
        def calculate_similarities(i):
            """Calculate similarities for row i of the matrix."""
            row_similarities = np.zeros(n)
            for j in range(i + 1, n):
                # Calculate cosine similarity
                if norms[i] > 0 and norms[j] > 0:  # Avoid division by zero
                    similarity = np.dot(embeddings[i], embeddings[j]) / (norms[i] * norms[j])
                else:
                    similarity = 0
                row_similarities[j] = similarity
            return i, row_similarities
        
        # Calculate similarities
        max_workers = min(os.cpu_count() * 2 or 4, n)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(calculate_similarities, i) for i in range(n)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    i, row_similarities = future.result()
                    # Fill in the similarity matrix (symmetric)
                    similarity_matrix[i, :] = row_similarities
                    for j in range(i + 1, n):
                        similarity_matrix[j, i] = row_similarities[j]
                except Exception as e:
                    logger.error(f"Error calculating similarities: {str(e)}")
        
        # Filter inputs
        filtered_indices = []
        for i in range(len(inputs)):
            # Check if this input is similar to any already selected input
            if not any(similarity_matrix[i, j] > similarity_threshold for j in filtered_indices):
                filtered_indices.append(i)
        
        # Get filtered inputs
        filtered_inputs = [inputs[i] for i in filtered_indices]
        
        logger.info(f"Removed {len(inputs) - len(filtered_inputs)} similar inputs")
        return filtered_inputs
    
    except Exception as e:
        logger.error(f"Failed to remove similar inputs: {str(e)}")
        return inputs

def check_document_relevance(
    document: Document,
    input: str,
    client: Any,
) -> bool:
    """
    Check if a document is relevant to an input.
    
    Args:
        document: The document to check.
        input: The input to check relevance for.
        client: The API client to use for embeddings.
        
    Returns:
        bool: True if the document is relevant, False otherwise.
    """
    logger.info(f"Checking relevance of document {document.id} for input: {input}")
    
    try:
        # Get embeddings
        doc_embedding = client.get_embeddings(document.content)[0]
        input_embedding = client.get_embeddings(input)[0]
        
        # Calculate similarity
        similarity = np.dot(doc_embedding, input_embedding) / (
            np.linalg.norm(doc_embedding) * np.linalg.norm(input_embedding)
        )
        
        # Check if document is relevant - using a lower threshold for debugging
        # Original threshold was 0.7, lowering to 0.3 for testing
        threshold = 0.3
        is_relevant = similarity > threshold
        
        logger.info(f"Document {document.id} relevance for input: {is_relevant} (similarity: {similarity:.4f}, threshold: {threshold})")
        return is_relevant
    
    except Exception as e:
        logger.error(f"Failed to check document relevance: {str(e)}")
        return False
