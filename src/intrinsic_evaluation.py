"""
Intrinsic Evaluation Framework

P.S. This was generated automatically and may contain errors. Please review carefully and rely on Jupyter notebooks for the original implementations.

A model-agnostic analytical framework for measuring encoder model performance
using pre-calculated embeddings. This module consolidates the evaluation methods
from the Demystifying Network Foundation Models paper.

Implemented methods:
- Cosine similarity metrics (intra-dataset anisotropy)
- Intrinsic dimensionality
- Metric alignment assessment (CKA with CIC features)
- Causal sensitivity testing
- Synth math (embedding arithmetic)

Author: Based on Demystifying Network Foundation Models (NeurIPS 2025)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm
import itertools
import warnings


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SynthDatasetEmbeddings:
    """Container for synthetic dataset embeddings used in causal sensitivity testing.
    
    Attributes:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: List of label strings identifying each sample's experimental condition
        label_mapping: Optional function to transform raw labels to standardized format
    """
    embeddings: np.ndarray
    labels: List[str]
    label_mapping: Optional[callable] = None
    
    def __post_init__(self):
        if self.label_mapping is not None:
            self.labels = [self.label_mapping(x) for x in self.labels]
        assert self.embeddings.shape[0] == len(self.labels), \
            f"Embeddings ({self.embeddings.shape[0]}) and labels ({len(self.labels)}) must have same length"


@dataclass
class DatasetInput:
    """Input container for a single dataset's embeddings and metadata.
    
    Attributes:
        dataset_name: Unique identifier for the dataset
        embeddings: Array of shape (n_samples, embedding_dim) containing model embeddings
        cic_features: Optional numpy array of CIC flowmeter features for metric alignment in the same order as embeddings
        cic_feature_names: Optional list of CIC feature column names
        synth_embeddings: Optional synthetic dataset embeddings for causal testing and math
    """
    dataset_name: str
    embeddings: np.ndarray
    cic_features: Optional[np.ndarray] = None
    cic_feature_names: Optional[List[str]] = None
    synth_embeddings: Optional[SynthDatasetEmbeddings] = None
    
    def __post_init__(self):
        # Ensure embeddings is a numpy array
        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = np.asarray(self.embeddings)
        
        # Clean NaN/Inf values
        self.embeddings = np.nan_to_num(self.embeddings, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class EvaluationResults:
    """Container for all evaluation results.
    
    Contains results from various evaluation methods organized by category.
    """
    # Cosine similarity metrics
    cosine_anisotropy: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Intrinsic dimensionality
    intrinsic_dimensionality: Optional[Dict[str, Tuple[float, float]]] = None
    
    # CKA alignment
    cka_with_cic_features: Optional[Dict[str, Dict[str, float]]] = None
    
    # Causal sensitivity
    causal_sensitivity: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Synth math
    synth_math_results: Optional[Dict[str, Dict[str, Any]]] = None
    
    def summary(self) -> str:
        """Generate a text summary of all results."""
        lines = ["=" * 60, "INTRINSIC EVALUATION FRAMEWORK RESULTS", "=" * 60]
        
        if self.cosine_anisotropy:
            lines.append("\n--- Cosine Anisotropy (Intra-Dataset) ---")
            for ds_name, metrics in self.cosine_anisotropy.items():
                lines.append(f"  {ds_name}: anisotropy={metrics['anisotropy']:.4f}")
        
        if self.intrinsic_dimensionality:
            lines.append("\n--- Intrinsic Dimensionality (2NN) ---")
            for ds_name, (id_val, id_err) in self.intrinsic_dimensionality.items():
                lines.append(f"  {ds_name}: ID={id_val:.2f} ± {id_err:.2f}")
        
        if self.cka_with_cic_features:
            lines.append("\n--- CKA with CIC Features ---")
            for ds_name, feature_ckas in self.cka_with_cic_features.items():
                avg_cka = np.mean(list(feature_ckas.values()))
                lines.append(f"  {ds_name}: avg_CKA={avg_cka:.4f} (over {len(feature_ckas)} features)")
        
        if self.causal_sensitivity:
            lines.append("\n--- Causal Sensitivity Testing ---")
            for ds_name, metrics in self.causal_sensitivity.items():
                lines.append(f"  {ds_name}:")
                if 'stability_baseline' in metrics:
                    lines.append(f"    Stability baseline: {metrics['stability_baseline']:.4f}")
                for cond, val in metrics.items():
                    if cond != 'stability_baseline' and isinstance(val, (int, float)):
                        lines.append(f"    {cond}: {val:.4f}")
        
        if self.synth_math_results:
            lines.append("\n--- Synth Math Results ---")
            for ds_name, results in self.synth_math_results.items():
                lines.append(f"  {ds_name}:")
                if 'target_cosine_similarity' in results:
                    lines.append(f"    Target cosine similarity: {results['target_cosine_similarity']:.4f}")
                if 'target_centroid_cosine' in results:
                    lines.append(f"    Target centroid cosine: {results['target_centroid_cosine']:.4f}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ============================================================================
# Internal CKA Functions (from cka.ipynb and cicflowmeter.ipynb)
# ============================================================================

def _debiased_dot_product_similarity_helper(
    xty: float,
    sum_squared_rows_x: np.ndarray,
    sum_squared_rows_y: np.ndarray,
    squared_norm_x: float,
    squared_norm_y: float,
    n: int
) -> float:
    """Helper for computing debiased dot product similarity (linear HSIC)."""
    return (
        xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def _feature_space_linear_cka(
    features_x: np.ndarray,
    features_y: np.ndarray,
    debiased: bool = False
) -> float:
    """Compute CKA with a linear kernel, in feature space.
    
    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.
    
    Args:
        features_x: A num_examples x num_features matrix of features.
        features_y: A num_examples x num_features matrix of features.
        debiased: Use unbiased estimator of dot product similarity.
        
    Returns:
        The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)
    
    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))
    
    if debiased:
        n = features_x.shape[0]
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)
        
        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n
        )
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n
        ))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n
        ))
    
    return dot_product_similarity / (normalization_x * normalization_y)


# ============================================================================
# Internal Cosine Similarity Functions (from cosine_anisotropy.ipynb)
# ============================================================================

def _cos_contrib(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute per-dimension cosine contribution between two embeddings.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        Array of per-dimension contributions to cosine similarity
    """
    numerator_terms = emb1 * emb2
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return np.array(numerator_terms / denom)


def _measure_anisotropy(embeddings: np.ndarray) -> Dict[str, Any]:
    """Measure anisotropy of embedding space.
    
    Computes the average cosine similarity between consecutive (randomly shuffled)
    embedding pairs and identifies dimensions contributing most to this similarity.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        
    Returns:
        Dictionary containing:
        - 'anisotropy': Estimated anisotropy score
        - 'top_dimensions': Top 10 dimensions contributing to anisotropy
        - 'top_contributions': Contribution values for top dimensions
    """
    # Shuffle embeddings
    indices = np.random.permutation(embeddings.shape[0])
    embeddings = embeddings[indices]
    
    layer_cosine_contribs = []
    
    for i in range(embeddings.shape[0] - 1):
        emb1, emb2 = embeddings[i, :], embeddings[i + 1, :]
        layer_cosine_contribs.append(_cos_contrib(emb1, emb2))
    
    layer_cosine_contribs = np.stack(layer_cosine_contribs)
    layer_cosine_contribs_mean = layer_cosine_contribs.mean(axis=0)
    
    aniso = layer_cosine_contribs_mean.sum()
    top_dims = np.argsort(layer_cosine_contribs_mean)[-10:]
    top_dims = np.flip(top_dims)
    
    return {
        'anisotropy': float(aniso),
        'top_dimensions': top_dims.tolist(),
        'top_contributions': [float(layer_cosine_contribs_mean[d]) for d in top_dims]
    }


def _measure_cosine_spread(
    arr1: np.ndarray,
    arr2: np.ndarray,
    exclude_self: bool = False
) -> float:
    """Compute mean pairwise cosine similarity between two sets of embeddings.
    
    Args:
        arr1: First set of embeddings (n1, dim)
        arr2: Second set of embeddings (n2, dim)
        exclude_self: If True and arrays are same object, exclude diagonal (self-similarity)
        
    Returns:
        Mean cosine similarity value
    """
    # Normalize vectors
    norm1 = np.linalg.norm(arr1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(arr2, axis=1, keepdims=True)
    arr1_normalized = arr1 / (norm1 + 1e-8)
    arr2_normalized = arr2 / (norm2 + 1e-8)
    
    # Compute cosine similarity matrix
    similarity_matrix = arr1_normalized @ arr2_normalized.T
    
    if exclude_self and arr1 is arr2:
        n = arr1.shape[0]
        mask = ~np.eye(n, dtype=bool)
        similarity_matrix = similarity_matrix[mask]
    
    return float(np.mean(similarity_matrix))


def _measure_l1_distance(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Compute mean pairwise L1 distance between two sets of embeddings.
    
    Args:
        arr1: First set of embeddings (n1, dim)
        arr2: Second set of embeddings (n2, dim)
        
    Returns:
        Mean L1 distance value
    """
    # Compute pairwise L1 distances using broadcasting
    diff = np.abs(arr1[:, np.newaxis, :] - arr2[np.newaxis, :, :])
    distances = diff.sum(axis=2)
    return float(np.mean(distances))


# ============================================================================
# Internal Intrinsic Dimensionality (from manifold_id_orig.ipynb)
# ============================================================================

def _calculate_intrinsic_dimensionality(
    embeddings: np.ndarray,
    maxk: int = 100
) -> Tuple[float, float]:
    """Calculate intrinsic dimensionality using 2-NN estimator.
    
    Uses the DADApy library for estimation.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        maxk: Maximum number of neighbors to consider for distance computation
        
    Returns:
        Tuple of (intrinsic_dimension, error_estimate)
    """
    try:
        from dadapy.data import Data
    except ImportError:
        raise ImportError(
            "dadapy is required for intrinsic dimensionality calculation. "
            "Install it with: pip install dadapy"
        )
    
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    dataset = Data(embeddings)
    dataset.remove_identical_points()
    dataset.compute_distances(maxk=min(maxk, embeddings.shape[0] - 1))
    
    ids = dataset.compute_id_2NN()
    
    # Returns (id_value, decimation_error, id_error)
    # We return (id_value, id_error) for simpler interface
    return (ids[0], ids[2])


# ============================================================================
# Internal CKA with CIC Features (from cicflowmeter.ipynb)
# ============================================================================

def _compute_cka_with_cic_features(
    embeddings: np.ndarray,
    cic_features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    show_progress: bool = True
) -> Dict[str, float]:
    """Compute CKA between embeddings and individual CIC flowmeter features.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        cic_features: Array of shape (n_samples, n_features) with CIC features
        feature_names: Optional list of feature names
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping feature names to CKA values
    """
    embeddings_np = embeddings
    n_features = cic_features.shape[1]
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    results = {}
    iterator = range(n_features)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing CKA with CIC features")
    
    for i in iterator:
        feature_values = cic_features[:, i].reshape(-1, 1)
        # Handle NaN/Inf in feature values
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            cka_val = _feature_space_linear_cka(embeddings_np, feature_values)
            results[feature_names[i]] = float(cka_val)
        except Exception as e:
            warnings.warn(f"Failed to compute CKA for feature {feature_names[i]}: {e}")
            results[feature_names[i]] = float('nan')
    
    return results


# ============================================================================
# Internal Causal Sensitivity Testing (from synthstability.ipynb)
# ============================================================================

def _linear_probing(
    embeddings: np.ndarray,
    labels: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[float, float]:
    """Perform linear probing to test separability of conditions.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: List of class labels
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_f1, test_f1) scores
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    
    X = embeddings
    y = np.array(labels)
    
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise ValueError("Linear probing requires exactly two classes")
    
    class_to_int = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_int = np.array([class_to_int[label] for label in y])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    return train_f1, test_f1


def _compute_causal_sensitivity(
    synth_data: SynthDatasetEmbeddings,
    baseline_label: str = 'fifo_6m_bbr_prof50_36_',
    conditions: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Compute causal sensitivity metrics from synthetic dataset embeddings.
    
    Tests how well the model distinguishes between different network conditions
    (congestion control, AQM, cross-traffic variations).
    
    Args:
        synth_data: SynthDatasetEmbeddings object containing embeddings and labels
        baseline_label: Label identifying the baseline condition
        conditions: Dict mapping condition labels to human-readable names.
                   If None, uses default mapping.
        
    Returns:
        Dictionary containing:
        - 'stability_baseline': Self-similarity of baseline condition
        - Per-condition cosine similarity and F1 scores
    """
    if conditions is None:
        conditions = {
            'fifo_6m_cubic_prof50_36_': "Congestion Control",
            'codel_6m_bbr_prof50_36_': "AQM",
            'fifo_6m_bbr_prof72_29_': "Crosstraffic",
            'codel_6m_cubic_prof72_29_': "All",
        }
    
    embeddings = synth_data.embeddings
    labels = synth_data.labels
    
    # Group embeddings by label
    classes = {}
    for label in set(labels):
        mask = np.array([l == label for l in labels], dtype=bool)
        classes[label] = embeddings[mask]
    
    results = {}
    
    # Baseline stability (self-similarity)
    if baseline_label in classes:
        baseline_emb = classes[baseline_label]
        results['stability_baseline'] = _measure_cosine_spread(
            baseline_emb, baseline_emb, exclude_self=True
        )
        results['stability_baseline_l1'] = _measure_l1_distance(baseline_emb, baseline_emb)
        
        # Compare to other conditions
        for cond_label, cond_name in conditions.items():
            if cond_label in classes:
                cond_emb = classes[cond_label]
                
                # Cosine similarity
                cos_sim = _measure_cosine_spread(baseline_emb, cond_emb)
                results[f'{cond_name}_cosine'] = cos_sim
                
                # L1 distance
                l1_dist = _measure_l1_distance(baseline_emb, cond_emb)
                results[f'{cond_name}_l1'] = l1_dist
                
                # Linear probing F1
                try:
                    combined_emb = np.concatenate([baseline_emb, cond_emb])
                    combined_labels = (
                        [baseline_label] * baseline_emb.shape[0] +
                        [cond_label] * cond_emb.shape[0]
                    )
                    _, test_f1 = _linear_probing(combined_emb, combined_labels)
                    results[f'{cond_name}_f1'] = test_f1
                except Exception as e:
                    warnings.warn(f"Linear probing failed for {cond_name}: {e}")
    else:
        warnings.warn(f"Baseline label '{baseline_label}' not found in data")
    
    return results


# ============================================================================
# Internal Synth Math (from math_embeddings.ipynb)
# ============================================================================

def _compute_synth_math(
    synth_data: SynthDatasetEmbeddings,
    label_order: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute embedding arithmetic on synthetic dataset.
    
    Tests whether: CC + AQM + CrossTraffic - 2*Baseline ≈ All
    
    Args:
        synth_data: SynthDatasetEmbeddings object
        label_order: List of labels [CC, AQM, CrossTraffic, Baseline, All]
                    If None, uses default ordering.
        
    Returns:
        Dictionary containing math test results
    """
    if label_order is None:
        label_order = [
            'fifo_6m_cubic_prof50_36',  # CC change
            'codel_6m_bbr_prof50_36',   # AQM change
            'fifo_6m_bbr_prof72_29',    # CrossTraffic change
            'fifo_6m_bbr_prof50_36',    # Baseline
            'codel_6m_cubic_prof72_29'  # All changes
        ]
    
    embeddings = synth_data.embeddings
    labels = synth_data.labels
    
    # Group embeddings by label
    classes = {}
    for label in set(labels):
        mask = np.array([l == label for l in labels], dtype=bool)
        classes[label] = embeddings[mask]
    
    results = {}
    
    # Check all required labels exist
    missing = [l for l in label_order if l not in classes]
    if missing:
        warnings.warn(f"Missing labels for synth math: {missing}")
        return results
    
    # Compute centroids
    centroids = {k: v.mean(axis=0) for k, v in classes.items()}
    
    # Compute intra-class spread (stability)
    mean_cos_sim = np.mean([
        _measure_cosine_spread(classes[label], classes[label], exclude_self=True)
        for label in classes
    ])
    mean_l1_dist = np.mean([
        _measure_l1_distance(classes[label], classes[label])
        for label in classes
    ])
    results['mean_intra_class_cosine'] = mean_cos_sim
    results['mean_intra_class_l1'] = mean_l1_dist
    
    # Baseline stability
    baseline_label = label_order[3]
    results['baseline_stability_cosine'] = _measure_cosine_spread(
        classes[baseline_label], classes[baseline_label], exclude_self=True
    )
    results['baseline_stability_l1'] = _measure_l1_distance(
        classes[baseline_label], classes[baseline_label]
    )
    
    # Pairwise distances between classes
    pairs = list(itertools.combinations(label_order, 2))
    for pair in pairs:
        cos_sim = _measure_cosine_spread(classes[pair[0]], classes[pair[1]])
        l1_dist = _measure_l1_distance(classes[pair[0]], classes[pair[1]])
        results[f'pairwise_{pair[0]}_vs_{pair[1]}_cosine'] = cos_sim
        results[f'pairwise_{pair[0]}_vs_{pair[1]}_l1'] = l1_dist
    
    # Math: CC + AQM + CrossTraffic - 2*Baseline
    cc_centroid = centroids[label_order[0]]
    aqm_centroid = centroids[label_order[1]]
    cross_centroid = centroids[label_order[2]]
    baseline_centroid = centroids[label_order[3]]
    target_centroid = centroids[label_order[4]]
    
    resulting_vec = cc_centroid + aqm_centroid + cross_centroid - 2 * baseline_centroid
    
    # Compute similarity to all classes
    for i, label in enumerate(label_order):
        cos_sim = _measure_cosine_spread(
            resulting_vec.reshape(1, -1), classes[label]
        )
        l1_dist = _measure_l1_distance(
            resulting_vec.reshape(1, -1), classes[label]
        )
        
        prefix = "target_" if i == 4 else f"class{i}_"
        results[f'{prefix}cosine_similarity'] = cos_sim
        results[f'{prefix}l1_distance'] = l1_dist
    
    # Also compute centroid-to-centroid comparison
    norm_result = np.linalg.norm(resulting_vec)
    norm_target = np.linalg.norm(target_centroid)
    target_cos = float(np.dot(resulting_vec, target_centroid) / (norm_result * norm_target + 1e-8))
    results['target_centroid_cosine'] = target_cos
    
    return results


# ============================================================================
# Main Evaluation Framework
# ============================================================================

class IntrinsicEvaluationFramework:
    """Main class for running the intrinsic evaluation framework.
    
    This framework evaluates encoder model embeddings using multiple metrics
    that don't require task-specific labels or fine-tuning.
    
    Example:
        >>> inputs = [
        ...     DatasetInput("dataset1", embeddings1, cic_features=cic1),
        ...     DatasetInput("dataset2", embeddings2, cic_features=cic2),
        ... ]
        >>> framework = IntrinsicEvaluationFramework()
        >>> results = framework.evaluate(inputs)
        >>> print(results.summary())
    """
    
    def __init__(
        self,
        compute_anisotropy: bool = True,
        compute_intrinsic_dim: bool = True,
        compute_cka_cic: bool = True,
        compute_causal_sensitivity: bool = True,
        compute_synth_math: bool = True,
        id_maxk: int = 100,
        verbose: bool = True
    ):
        """Initialize the evaluation framework.
        
        Args:
            compute_anisotropy: Whether to compute cosine anisotropy metrics
            compute_intrinsic_dim: Whether to compute intrinsic dimensionality
            compute_cka_cic: Whether to compute CKA with CIC features
            compute_causal_sensitivity: Whether to compute causal sensitivity
            compute_synth_math: Whether to compute synth math tests
            id_maxk: Max neighbors for intrinsic dimensionality
            verbose: Whether to print progress information
        """
        self.compute_anisotropy = compute_anisotropy
        self.compute_intrinsic_dim = compute_intrinsic_dim
        self.compute_cka_cic = compute_cka_cic
        self.compute_causal_sensitivity = compute_causal_sensitivity
        self.compute_synth_math = compute_synth_math
        self.id_maxk = id_maxk
        self.verbose = verbose
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def evaluate(
        self,
        datasets: List[DatasetInput]
    ) -> EvaluationResults:
        """Run all enabled evaluations on the provided datasets.
        
        Args:
            datasets: List of DatasetInput objects to evaluate
        
        Returns:
            EvaluationResults object containing all computed metrics
        """
        results = EvaluationResults()
        
        # 1. Cosine Anisotropy (per dataset)
        if self.compute_anisotropy:
            self._log("Computing cosine anisotropy...")
            results.cosine_anisotropy = {}
            for ds in tqdm(datasets, desc="Anisotropy", disable=not self.verbose):
                results.cosine_anisotropy[ds.dataset_name] = _measure_anisotropy(
                    ds.embeddings
                )
        
        # 2. Intrinsic Dimensionality
        if self.compute_intrinsic_dim:
            self._log("Computing intrinsic dimensionality...")
            results.intrinsic_dimensionality = {}
            for ds in tqdm(datasets, desc="ID", disable=not self.verbose):
                try:
                    id_result = _calculate_intrinsic_dimensionality(
                        ds.embeddings, maxk=self.id_maxk
                    )
                    results.intrinsic_dimensionality[ds.dataset_name] = id_result
                except Exception as e:
                    warnings.warn(f"ID calculation failed for {ds.dataset_name}: {e}")
        
        # 3. CKA with CIC features (per dataset, only if cic_features provided)
        if self.compute_cka_cic:
            self._log("Computing CKA with CIC features...")
            results.cka_with_cic_features = {}
            for ds in datasets:
                if ds.cic_features is not None:
                    cka_results = _compute_cka_with_cic_features(
                        ds.embeddings,
                        ds.cic_features,
                        ds.cic_feature_names,
                        show_progress=self.verbose
                    )
                    results.cka_with_cic_features[ds.dataset_name] = cka_results
        
        # 4. Causal Sensitivity Testing
        if self.compute_causal_sensitivity:
            self._log("Computing causal sensitivity...")
            results.causal_sensitivity = {}
            for ds in datasets:
                if ds.synth_embeddings is not None:
                    sens_results = _compute_causal_sensitivity(ds.synth_embeddings)
                    results.causal_sensitivity[ds.dataset_name] = sens_results
        
        # 5. Synth Math
        if self.compute_synth_math:
            self._log("Computing synth math...")
            results.synth_math_results = {}
            
            for ds in datasets:
                if ds.synth_embeddings is not None:
                    math_results = _compute_synth_math(ds.synth_embeddings)
                    results.synth_math_results[ds.dataset_name] = math_results
        
        self._log("Evaluation complete!")
        return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example usage with synthetic data
    print("Intrinsic Evaluation Framework - Example")
    print("=" * 50)
    
    # Create synthetic test data
    n_samples = 100
    embedding_dim = 768
    
    # Random embeddings for demonstration
    np.random.seed(42)
    emb1 = np.random.randn(n_samples, embedding_dim)
    emb2 = np.random.randn(n_samples, embedding_dim)
    
    # Create dataset inputs
    datasets = [
        DatasetInput(
            dataset_name="test_dataset_1",
            embeddings=emb1
        ),
        DatasetInput(
            dataset_name="test_dataset_2",
            embeddings=emb2
        ),
    ]
    
    # Initialize framework (disable ID calculation if dadapy not installed)
    try:
        from dadapy.data import Data
        has_dadapy = True
    except ImportError:
        has_dadapy = False
        print("Note: dadapy not installed, skipping intrinsic dimensionality")
    
    framework = IntrinsicEvaluationFramework(
        compute_anisotropy=True,
        compute_intrinsic_dim=has_dadapy,
        compute_cka_cic=False,  # No CIC features in example
        compute_causal_sensitivity=False,  # No synth data in example
        compute_synth_math=False,  # No synth data in example
        verbose=True
    )
    
    # Run evaluation
    results = framework.evaluate(datasets)
    
    # Print results
    print(results.summary())
