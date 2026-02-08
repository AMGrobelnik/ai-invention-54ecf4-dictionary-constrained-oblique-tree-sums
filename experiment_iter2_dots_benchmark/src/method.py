#!/usr/bin/env python3
"""
DOTS: Dictionary-Constrained Oblique Tree Sums
Full Benchmark Experiment with K-Sweep and Stability Analysis

Implements:
1. FIGS-style greedy competitive tree growth (axis-aligned + oblique)
2. DOTS: dictionary-constrained oblique splits with alternating optimization
3. Baselines: RandomForest, DecisionTree, LogisticRegression (FIGS axis-aligned)
4. K-sweep analysis for DOTS (K=2,3,4,5,6,8,10)
5. Dictionary stability via 5-fold CV with Hungarian matching

Usage:
    .venv/bin/python method.py                    # Run on full data (200 examples)
    .venv/bin/python method.py --max-examples 10  # Run on first 10 examples
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Logging setup — DEBUG level as required
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dots")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_SAMPLES_LEAF = 5
RANDOM_SEED = 42

DATA_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/"
    "run__20260208_015300/3_invention_loop/iter_1/gen_art/"
    "data_id1_it1__opus"
)
WORKSPACE = Path(__file__).parent


# ===========================================================================
# Section 1: Data Loading
# ===========================================================================
def load_data(filepath: str, max_examples: int = None):
    """Load JSON dataset, extract features/labels, separate train/test.

    If all examples share the same split label, creates a stratified
    80/20 train/test split automatically.
    """
    logger.info(f"Loading data from {filepath}")
    try:
        raw = json.loads(Path(filepath).read_text())
    except Exception as e:
        logger.error(f"Failed to read data file: {e}")
        raise

    examples = raw if isinstance(raw, list) else raw.get("examples", raw)
    logger.debug(f"Raw examples count: {len(examples)}")

    if max_examples is not None and max_examples < len(examples):
        logger.info(f"Limiting to first {max_examples} examples")
        examples = examples[:max_examples]

    feature_names = None
    all_X, all_y = [], []

    for idx, ex in enumerate(examples):
        try:
            ctx = ex["context"]
            features_dict = ctx["features"]
            if feature_names is None:
                feature_names = list(features_dict.keys())
            x_row = [features_dict[fn] for fn in feature_names]
            y_val = int(ex["output"])
            all_X.append(x_row)
            all_y.append(y_val)
        except KeyError as e:
            logger.error(f"Example {idx} missing key: {e}")
            raise

    all_X = np.array(all_X, dtype=np.float64)
    all_y = np.array(all_y, dtype=np.float64)
    logger.debug(f"Feature matrix shape: {all_X.shape}, labels shape: {all_y.shape}")

    # Check if data already has a meaningful train/test split
    splits = [ex["split"] for ex in examples]
    unique_splits = set(splits)
    logger.debug(f"Unique split labels: {unique_splits}")

    if len(unique_splits) > 1 and "train" in unique_splits:
        train_mask = np.array([s == "train" for s in splits])
        test_mask = ~train_mask
        logger.info("Using existing train/test split from data")
    else:
        # AIDEV-NOTE: All examples share one split label → create 80/20 stratified split
        logger.info("No train/test split found — creating stratified 80/20 split")
        from sklearn.model_selection import train_test_split as tts

        np.random.seed(RANDOM_SEED)
        indices = np.arange(len(examples))
        unique_classes, counts = np.unique(all_y, return_counts=True)
        can_stratify = all(c >= 2 for c in counts) and len(examples) >= 5
        logger.debug(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")
        logger.debug(f"Can stratify: {can_stratify}")

        try:
            tr_idx, te_idx = tts(
                indices,
                test_size=max(0.2, 1.0 / len(examples)),
                stratify=all_y if can_stratify else None,
                random_state=RANDOM_SEED,
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}), falling back to random")
            n_test = max(1, len(examples) // 5)
            rng = np.random.RandomState(RANDOM_SEED)
            perm = rng.permutation(len(examples))
            te_idx = perm[:n_test]
            tr_idx = perm[n_test:]

        train_mask = np.zeros(len(examples), dtype=bool)
        train_mask[tr_idx] = True
        test_mask = ~train_mask
        for i in range(len(examples)):
            examples[i]["split"] = "train" if train_mask[i] else "test"

    train_indices = np.where(train_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()

    X_train = all_X[train_mask]
    y_train = all_y[train_mask]
    X_test = all_X[test_mask]
    y_test = all_y[test_mask]

    logger.info(
        f"Data loaded: {len(X_train)} train, {len(X_test)} test, "
        f"{len(feature_names)} features, "
        f"class balance={y_train.mean():.3f} positive"
    )

    # Sanity checks
    assert len(X_train) > 0, "No training examples!"
    assert len(X_test) > 0, "No test examples!"
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch!"
    assert not np.any(np.isnan(X_train)), "NaN in training features!"
    assert not np.any(np.isnan(X_test)), "NaN in test features!"

    return (
        X_train, y_train, X_test, y_test,
        feature_names, examples, train_indices, test_indices,
    )


def standardize(X_train, X_test):
    """Z-score standardization fitted on train only."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    X_tr_s = (X_train - mu) / sigma
    X_te_s = (X_test - mu) / sigma
    logger.debug(
        f"Standardized: train mean={X_tr_s.mean():.4f}, "
        f"std={X_tr_s.std():.4f}"
    )
    return X_tr_s, X_te_s, mu, sigma


# ===========================================================================
# Section 2: Core Data Structures
# ===========================================================================
class LeafNode:
    __slots__ = ("value", "n_samples")

    def __init__(self, value: float, n_samples: int):
        self.value = value
        self.n_samples = n_samples


class SplitNode:
    __slots__ = (
        "direction_index", "direction_vector", "threshold",
        "left", "right", "n_samples",
    )

    def __init__(self, direction_index, direction_vector, threshold,
                 left, right, n_samples):
        self.direction_index = direction_index
        self.direction_vector = direction_vector
        self.threshold = threshold
        self.left = left
        self.right = right
        self.n_samples = n_samples


def _traverse(node, x):
    if isinstance(node, LeafNode):
        return node.value
    proj = np.dot(node.direction_vector, x)
    return _traverse(node.left, x) if proj <= node.threshold else _traverse(node.right, x)


class TreeModel:
    def __init__(self, root):
        self.root = root

    def predict(self, X):
        return np.array([_traverse(self.root, X[i]) for i in range(len(X))])


def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class FIGSEnsemble:
    def __init__(self, trees, intercept):
        self.trees = trees
        self.intercept = intercept

    def predict_raw(self, X):
        raw = np.full(len(X), self.intercept)
        for tree in self.trees:
            raw += tree.predict(X)
        return raw

    def predict_proba(self, X):
        return _sigmoid(self.predict_raw(X))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


# ===========================================================================
# Section 3: Split Finding & Greedy Growth
# ===========================================================================
def _best_threshold(projections, residuals):
    """Find best threshold for 1-D projections maximizing variance reduction."""
    n = len(projections)
    if n < 2 * MIN_SAMPLES_LEAF:
        return None, -np.inf

    order = np.argsort(projections)
    proj_sorted = projections[order]
    res_sorted = residuals[order]

    total_sum = res_sorted.sum()
    total_sq = (res_sorted ** 2).sum()
    total_var = total_sq - total_sum ** 2 / n

    left_sum = 0.0
    left_sq = 0.0
    best_gain = -np.inf
    best_threshold = None

    for i in range(MIN_SAMPLES_LEAF - 1, n - MIN_SAMPLES_LEAF):
        left_sum += res_sorted[i]
        left_sq += res_sorted[i] ** 2
        if proj_sorted[i] == proj_sorted[i + 1]:
            continue
        n_left = i + 1
        n_right = n - n_left
        right_sum = total_sum - left_sum
        left_var = left_sq - left_sum ** 2 / n_left
        right_var = (total_sq - left_sq) - right_sum ** 2 / n_right
        gain = total_var - left_var - right_var
        if gain > best_gain:
            best_gain = gain
            best_threshold = (proj_sorted[i] + proj_sorted[i + 1]) / 2.0

    return best_threshold, best_gain


def find_best_split(X, residuals, mode, dictionary=None):
    """Find the best split for data at a node."""
    n_samples, n_features = X.shape
    best = {"direction_vector": None, "direction_index": -1,
            "threshold": None, "gain": -np.inf}

    try:
        if mode == "axis_aligned":
            for j in range(n_features):
                thr, gain = _best_threshold(X[:, j], residuals)
                if gain > best["gain"]:
                    dv = np.zeros(n_features)
                    dv[j] = 1.0
                    best = {"direction_vector": dv, "direction_index": j,
                            "threshold": thr, "gain": gain}

        elif mode == "dots":
            K = len(dictionary)
            for k in range(K):
                proj = X @ dictionary[k]
                thr, gain = _best_threshold(proj, residuals)
                if gain > best["gain"]:
                    best = {"direction_vector": dictionary[k].copy(),
                            "direction_index": k, "threshold": thr, "gain": gain}

        elif mode == "oblique_unconstrained":
            candidates = []
            # axis-aligned
            for j in range(n_features):
                dv = np.zeros(n_features)
                dv[j] = 1.0
                candidates.append(dv)
            # random projections
            rng = np.random.RandomState(RANDOM_SEED)
            for _ in range(20):
                rv = rng.randn(n_features)
                rv /= np.linalg.norm(rv) + 1e-12
                candidates.append(rv)
            # PCA directions
            if n_samples >= 3:
                n_comp = min(3, n_features, n_samples)
                pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
                pca.fit(X)
                for comp in pca.components_:
                    candidates.append(comp / (np.linalg.norm(comp) + 1e-12))

            for dv in candidates:
                thr, gain = _best_threshold(X @ dv, residuals)
                if gain > best["gain"]:
                    best = {"direction_vector": dv.copy(), "direction_index": -1,
                            "threshold": thr, "gain": gain}

            # Coordinate descent refinement
            if best["direction_vector"] is not None:
                cur_dv = best["direction_vector"].copy()
                for _ in range(5):
                    improved = False
                    for j in range(n_features):
                        for delta in [-0.1, 0.1, -0.01, 0.01]:
                            trial = cur_dv.copy()
                            trial[j] += delta
                            trial /= np.linalg.norm(trial) + 1e-12
                            thr, gain = _best_threshold(X @ trial, residuals)
                            if gain > best["gain"]:
                                best["direction_vector"] = trial.copy()
                                best["threshold"] = thr
                                best["gain"] = gain
                                cur_dv = trial.copy()
                                improved = True
                    if not improved:
                        break
    except Exception as e:
        logger.error(f"Error in find_best_split (mode={mode}): {e}")
        logger.debug(traceback.format_exc())

    return best


def _newton_leaf_value(residuals, proba, shrinkage=0.1):
    """Newton-Raphson leaf value for log-loss boosting."""
    hessian = proba * (1.0 - proba) + 1e-8
    return shrinkage * residuals.sum() / hessian.sum()


def grow_figs_ensemble(X_train, y_train, mode, max_splits=15,
                       max_trees=5, dictionary=None, shrinkage=0.3):
    """Grow a FIGS-style competitive ensemble of trees."""
    n = len(y_train)
    if n == 0:
        logger.warning("Empty training set, returning empty ensemble")
        return FIGSEnsemble(trees=[], intercept=0.0)

    p = np.clip(y_train.mean(), 0.01, 0.99)
    intercept = float(np.log(p / (1.0 - p)))
    logger.debug(
        f"grow_figs: mode={mode}, max_splits={max_splits}, max_trees={max_trees}, "
        f"shrinkage={shrinkage}, n={n}, intercept={intercept:.4f}"
    )

    trees = []
    leaf_registry = []

    for split_iter in range(max_splits):
        preds = np.full(n, intercept)
        for tree in trees:
            preds += tree.predict(X_train)
        proba = _sigmoid(preds)
        residuals = y_train - proba

        candidates = []

        # Option A: extend existing leaf
        for leaf_info in leaf_registry:
            tree_idx, _, indices, _ = leaf_info
            if len(indices) < 2 * MIN_SAMPLES_LEAF:
                continue
            split_info = find_best_split(
                X=X_train[indices], residuals=residuals[indices],
                mode=mode, dictionary=dictionary,
            )
            if split_info["gain"] > 0:
                candidates.append((split_info["gain"], "extend", leaf_info, split_info))

        # Option B: new tree
        if len(trees) < max_trees:
            all_idx = np.arange(n)
            split_info = find_best_split(
                X=X_train, residuals=residuals, mode=mode, dictionary=dictionary,
            )
            if split_info["gain"] > 0:
                candidates.append((split_info["gain"], "new_tree", all_idx, split_info))

        if not candidates:
            logger.debug(f"  No more valid splits at iteration {split_iter}")
            break

        candidates.sort(key=lambda c: c[0], reverse=True)
        _, action, info, split_info = candidates[0]
        dv = split_info["direction_vector"]
        di = split_info["direction_index"]
        thr = split_info["threshold"]

        if action == "new_tree":
            all_idx = info
            proj = X_train[all_idx] @ dv
            left_mask = proj <= thr
            right_mask = ~left_mask
            left_idx = all_idx[left_mask]
            right_idx = all_idx[right_mask]
            if len(left_idx) == 0 or len(right_idx) == 0:
                logger.debug(f"  Empty child at split {split_iter}, stopping")
                break

            left_leaf = LeafNode(
                value=_newton_leaf_value(residuals[left_idx], proba[left_idx], shrinkage),
                n_samples=len(left_idx),
            )
            right_leaf = LeafNode(
                value=_newton_leaf_value(residuals[right_idx], proba[right_idx], shrinkage),
                n_samples=len(right_idx),
            )
            root = SplitNode(di, dv, thr, left_leaf, right_leaf, len(all_idx))
            new_tree = TreeModel(root)
            trees.append(new_tree)
            tree_idx = len(trees) - 1
            leaf_registry.append((tree_idx, left_leaf, left_idx, (root, "left")))
            leaf_registry.append((tree_idx, right_leaf, right_idx, (root, "right")))

        elif action == "extend":
            leaf_info = info
            tree_idx, old_leaf, indices, (parent_node, side) = leaf_info
            leaf_registry.remove(leaf_info)

            proj = X_train[indices] @ dv
            left_mask = proj <= thr
            right_mask = ~left_mask
            left_idx = indices[left_mask]
            right_idx = indices[right_mask]
            if len(left_idx) == 0 or len(right_idx) == 0:
                leaf_registry.append(leaf_info)
                continue

            left_leaf = LeafNode(
                value=_newton_leaf_value(residuals[left_idx], proba[left_idx], shrinkage),
                n_samples=len(left_idx),
            )
            right_leaf = LeafNode(
                value=_newton_leaf_value(residuals[right_idx], proba[right_idx], shrinkage),
                n_samples=len(right_idx),
            )
            new_node = SplitNode(di, dv, thr, left_leaf, right_leaf, len(indices))
            if side == "left":
                parent_node.left = new_node
            else:
                parent_node.right = new_node
            leaf_registry.append((tree_idx, left_leaf, left_idx, (new_node, "left")))
            leaf_registry.append((tree_idx, right_leaf, right_idx, (new_node, "right")))

    # Refit leaf values
    if trees:
        preds = np.full(n, intercept)
        for tree in trees:
            preds += tree.predict(X_train)
        proba_final = _sigmoid(preds)
        residuals_final = y_train - proba_final
        for leaf_info in leaf_registry:
            _, leaf, indices, _ = leaf_info
            if len(indices) > 0:
                leaf.value = _newton_leaf_value(
                    residuals_final[indices], proba_final[indices], shrinkage,
                )

    logger.debug(f"  Built {len(trees)} trees with {sum(_count_splits(t.root) for t in trees)} total splits")
    return FIGSEnsemble(trees=trees, intercept=intercept)


# ===========================================================================
# Section 4: DOTS Dictionary Init & Alternating Optimization
# ===========================================================================
def initialize_dictionary(X, K):
    """Initialize K dictionary directions using PCA."""
    n_samples, n_features = X.shape
    n_comp = min(K, n_features, n_samples)
    logger.debug(f"Initializing dictionary: K={K}, n_comp={n_comp}")
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    pca.fit(X)
    dictionary = np.zeros((K, n_features))
    for i in range(n_comp):
        v = pca.components_[i]
        dictionary[i] = v / (np.linalg.norm(v) + 1e-12)
    rng = np.random.RandomState(RANDOM_SEED + 1)
    for i in range(n_comp, K):
        rv = rng.randn(n_features)
        dictionary[i] = rv / (np.linalg.norm(rv) + 1e-12)
    logger.debug(f"Dictionary shape: {dictionary.shape}, norms: {np.linalg.norm(dictionary, axis=1)[:3]}...")
    return dictionary


def _collect_splits(node):
    if isinstance(node, LeafNode):
        return []
    result = [node]
    result.extend(_collect_splits(node.left))
    result.extend(_collect_splits(node.right))
    return result


def _compute_log_loss(ensemble, X, y):
    proba = np.clip(ensemble.predict_proba(X), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba)))


def optimize_dictionary(ensemble, X_train, y_train, dictionary,
                        n_steps=5, lr=0.005, epsilon=1e-4):
    """Optimize dictionary directions via finite-difference gradient descent."""
    K, n_features = dictionary.shape
    dictionary = dictionary.copy()

    used_indices = set()
    for tree in ensemble.trees:
        for node in _collect_splits(tree.root):
            if node.direction_index >= 0:
                used_indices.add(node.direction_index)
    logger.debug(f"  Dictionary optimization: used indices={used_indices}")

    for k in range(K):
        if k not in used_indices:
            continue
        for step in range(n_steps):
            _update_ensemble_directions(ensemble, dictionary)
            grad = np.zeros(n_features)
            for j in range(n_features):
                dictionary[k, j] += epsilon
                _update_ensemble_directions(ensemble, dictionary)
                loss_plus = _compute_log_loss(ensemble, X_train, y_train)
                dictionary[k, j] -= 2 * epsilon
                _update_ensemble_directions(ensemble, dictionary)
                loss_minus = _compute_log_loss(ensemble, X_train, y_train)
                dictionary[k, j] += epsilon
                grad[j] = (loss_plus - loss_minus) / (2 * epsilon)
            dictionary[k] -= lr * grad
            norm = np.linalg.norm(dictionary[k])
            if norm > 1e-12:
                dictionary[k] /= norm

    _update_ensemble_directions(ensemble, dictionary)
    return dictionary


def _update_ensemble_directions(ensemble, dictionary):
    for tree in ensemble.trees:
        for node in _collect_splits(tree.root):
            idx = node.direction_index
            if 0 <= idx < len(dictionary):
                node.direction_vector = dictionary[idx].copy()


def dots_full(X_train, y_train, K, max_splits=15, max_trees=5,
              n_alternation_rounds=3):
    """Run full DOTS algorithm with alternating optimization."""
    logger.debug(f"dots_full: K={K}, max_splits={max_splits}, rounds={n_alternation_rounds}")
    dictionary = initialize_dictionary(X_train, K)
    ensemble = None

    for round_idx in range(n_alternation_rounds):
        ensemble = grow_figs_ensemble(
            X_train=X_train, y_train=y_train, mode="dots",
            max_splits=max_splits, max_trees=max_trees, dictionary=dictionary,
        )
        if round_idx < n_alternation_rounds - 1:
            try:
                loss_before = _compute_log_loss(ensemble, X_train, y_train)
                dictionary = optimize_dictionary(
                    ensemble=ensemble, X_train=X_train, y_train=y_train,
                    dictionary=dictionary, n_steps=5, lr=0.005,
                )
                loss_after = _compute_log_loss(ensemble, X_train, y_train)
                logger.debug(
                    f"  DOTS K={K} round {round_idx}: "
                    f"loss {loss_before:.4f} -> {loss_after:.4f}"
                )
            except Exception as e:
                logger.warning(f"Dictionary optimization failed at round {round_idx}: {e}")
                # Fallback: keep current dictionary unchanged
                break

    return ensemble, dictionary


# ===========================================================================
# Section 5: Baselines
# ===========================================================================
def run_baseline_rf(X_train, y_train, X_test):
    logger.debug("Running RandomForest baseline")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_train)
    test_pred = rf.predict(X_test)
    test_proba = rf.predict_proba(X_test)[:, 1]
    return train_pred, test_pred, test_proba


def run_baseline_dt(X_train, y_train, X_test):
    logger.debug("Running DecisionTree baseline")
    dt = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED)
    dt.fit(X_train, y_train)
    train_pred = dt.predict(X_train)
    test_pred = dt.predict(X_test)
    test_proba = dt.predict_proba(X_test)[:, 1]
    return train_pred, test_pred, test_proba


def run_baseline_lr(X_train, y_train, X_test):
    logger.debug("Running LogisticRegression baseline")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_train, y_train)
    train_pred = lr.predict(X_train)
    test_pred = lr.predict(X_test)
    test_proba = lr.predict_proba(X_test)[:, 1]
    return train_pred, test_pred, test_proba


# ===========================================================================
# Section 6: Evaluation Utilities
# ===========================================================================
def _count_splits(node):
    if isinstance(node, LeafNode):
        return 0
    return 1 + _count_splits(node.left) + _count_splits(node.right)


def count_total_splits(ensemble):
    return sum(_count_splits(t.root) for t in ensemble.trees)


def _collect_direction_indices(node):
    if isinstance(node, LeafNode):
        return []
    result = [node.direction_index]
    result.extend(_collect_direction_indices(node.left))
    result.extend(_collect_direction_indices(node.right))
    return result


def count_unique_directions(ensemble):
    all_dirs = []
    for tree in ensemble.trees:
        all_dirs.extend(_collect_direction_indices(tree.root))
    return len(set(all_dirs))


def evaluate_ensemble(ensemble, X_train, y_train, X_test, y_test):
    """Evaluate a FIGSEnsemble and return metrics dict."""
    try:
        train_pred = ensemble.predict(X_train)
        test_pred = ensemble.predict(X_test)
        train_proba = ensemble.predict_proba(X_train)
        test_proba = ensemble.predict_proba(X_test)

        result = {
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "train_predictions": train_pred.tolist(),
            "test_predictions": test_pred.tolist(),
            "test_probabilities": test_proba.tolist(),
        }
        # AUROC needs both classes present
        if len(np.unique(y_test)) > 1 and len(np.unique(test_proba)) > 1:
            result["test_auroc"] = float(roc_auc_score(y_test, test_proba))
        else:
            result["test_auroc"] = 0.5
            logger.warning("AUROC undefined (single class or constant proba), defaulting to 0.5")

        if len(np.unique(y_train)) > 1 and len(np.unique(train_proba)) > 1:
            result["train_auroc"] = float(roc_auc_score(y_train, train_proba))
        else:
            result["train_auroc"] = 0.5

        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.debug(traceback.format_exc())
        raise


def compute_metrics(train_pred, test_pred, test_proba, y_train, y_test):
    """Compute metrics for sklearn baselines."""
    result = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "train_predictions": train_pred.tolist(),
        "test_predictions": test_pred.tolist(),
        "test_probabilities": test_proba.tolist(),
    }
    if len(np.unique(y_test)) > 1 and len(np.unique(test_proba)) > 1:
        result["test_auroc"] = float(roc_auc_score(y_test, test_proba))
    else:
        result["test_auroc"] = 0.5
    return result


def name_directions(dictionary, feature_names, top_k=3):
    names = []
    for k in range(len(dictionary)):
        w = dictionary[k]
        abs_w = np.abs(w)
        top_idx = np.argsort(abs_w)[-top_k:][::-1]
        parts = []
        for idx in top_idx:
            if abs_w[idx] > 0.05:
                sign = "+" if w[idx] > 0 else "-"
                parts.append(f"{sign}{abs_w[idx]:.2f}*{feature_names[idx]}")
        name = f"Concept_{k+1}: {' '.join(parts)}" if parts else f"Concept_{k+1}: uniform"
        names.append(name)
    return names


def compute_dictionary_stability(fold_dictionaries):
    """Compute pairwise cosine similarity between fold dictionaries."""
    pairwise_sims = []
    try:
        for i in range(len(fold_dictionaries)):
            for j in range(i + 1, len(fold_dictionaries)):
                cos_matrix = np.abs(fold_dictionaries[i] @ fold_dictionaries[j].T)
                row_ind, col_ind = linear_sum_assignment(-cos_matrix)
                matched = cos_matrix[row_ind, col_ind]
                pairwise_sims.append(float(np.mean(matched)))
    except Exception as e:
        logger.warning(f"Stability computation failed: {e}")
        # Greedy fallback
        for i in range(len(fold_dictionaries)):
            for j in range(i + 1, len(fold_dictionaries)):
                d_i, d_j = fold_dictionaries[i], fold_dictionaries[j]
                cos_matrix = np.abs(d_i @ d_j.T)
                sims = []
                used = set()
                for row in range(len(d_i)):
                    best_col = -1
                    best_sim = -1
                    for col in range(len(d_j)):
                        if col not in used and cos_matrix[row, col] > best_sim:
                            best_sim = cos_matrix[row, col]
                            best_col = col
                    if best_col >= 0:
                        used.add(best_col)
                        sims.append(best_sim)
                pairwise_sims.append(float(np.mean(sims)) if sims else 0.0)

    return {
        "mean_cosine": float(np.mean(pairwise_sims)) if pairwise_sims else 0.0,
        "pairwise": pairwise_sims,
    }


# ===========================================================================
# Section 7: Full Experiment Pipeline
# ===========================================================================
def run_full_experiment(data_filepath, max_examples=None):
    """Run the complete DOTS experiment."""
    t0 = time.time()
    np.random.seed(RANDOM_SEED)

    # 1. Load data
    (X_train, y_train, X_test, y_test,
     feature_names, raw_examples, train_indices, test_indices) = load_data(
        data_filepath, max_examples=max_examples,
    )

    # 2. Standardize
    X_train_s, X_test_s, mu, sigma = standardize(X_train, X_test)

    results = {}
    n_train = len(X_train_s)

    # Adaptive hyperparameters based on dataset size
    figs_max_splits = min(25, max(5, n_train // 6))
    dots_max_splits = min(15, max(5, n_train // 10))
    exp_max_trees = 5
    logger.info(
        f"Hyperparameters: figs_splits={figs_max_splits}, "
        f"dots_splits={dots_max_splits}, max_trees={exp_max_trees}"
    )

    # 3a. Axis-aligned FIGS baseline
    logger.info("Running axis-aligned FIGS baseline...")
    try:
        figs_aa = grow_figs_ensemble(
            X_train_s, y_train, mode="axis_aligned",
            max_splits=figs_max_splits, max_trees=exp_max_trees,
            shrinkage=1.0,
        )
        results["figs_axis_aligned"] = evaluate_ensemble(
            figs_aa, X_train_s, y_train, X_test_s, y_test,
        )
        results["figs_axis_aligned"]["n_unique_directions"] = count_unique_directions(figs_aa)
        results["figs_axis_aligned"]["n_splits"] = count_total_splits(figs_aa)
        logger.info(f"  FIGS AA: acc={results['figs_axis_aligned']['test_accuracy']:.4f}, "
                     f"auroc={results['figs_axis_aligned']['test_auroc']:.4f}")
    except Exception as e:
        logger.error(f"FIGS axis-aligned failed: {e}")
        logger.debug(traceback.format_exc())
        raise

    # 3b. Unconstrained oblique FIGS
    logger.info("Running unconstrained oblique FIGS...")
    try:
        figs_ob = grow_figs_ensemble(
            X_train_s, y_train, mode="oblique_unconstrained",
            max_splits=figs_max_splits, max_trees=exp_max_trees,
            shrinkage=1.0,
        )
        results["figs_oblique"] = evaluate_ensemble(
            figs_ob, X_train_s, y_train, X_test_s, y_test,
        )
        results["figs_oblique"]["n_unique_directions"] = count_unique_directions(figs_ob)
        results["figs_oblique"]["n_splits"] = count_total_splits(figs_ob)
        logger.info(f"  FIGS oblique: acc={results['figs_oblique']['test_accuracy']:.4f}, "
                     f"auroc={results['figs_oblique']['test_auroc']:.4f}")
    except Exception as e:
        logger.error(f"FIGS oblique failed: {e}")
        logger.debug(traceback.format_exc())
        raise

    # 3c. DOTS K-sweep
    K_values = [2, 3, 4, 5, 6, 8, 10]
    for K in K_values:
        logger.info(f"Running DOTS K={K}...")
        try:
            ens, dct = dots_full(
                X_train_s, y_train, K=K,
                max_splits=dots_max_splits, max_trees=exp_max_trees,
                n_alternation_rounds=3,
            )
            key = f"dots_K{K}"
            results[key] = evaluate_ensemble(ens, X_train_s, y_train, X_test_s, y_test)
            results[key]["K"] = K
            results[key]["n_unique_directions"] = K
            results[key]["n_splits"] = count_total_splits(ens)
            results[key]["dictionary"] = dct.tolist()
            results[key]["dictionary_feature_names"] = feature_names
            results[key]["direction_names"] = name_directions(dct, feature_names)
            logger.info(f"  DOTS K={K}: acc={results[key]['test_accuracy']:.4f}, "
                         f"auroc={results[key]['test_auroc']:.4f}")
        except Exception as e:
            logger.error(f"DOTS K={K} failed: {e}")
            logger.debug(traceback.format_exc())

    # 3d. Sklearn baselines
    logger.info("Running sklearn baselines...")
    try:
        rf_tr, rf_te, rf_proba = run_baseline_rf(X_train_s, y_train, X_test_s)
        results["random_forest"] = compute_metrics(rf_tr, rf_te, rf_proba, y_train, y_test)
        logger.info(f"  RF: acc={results['random_forest']['test_accuracy']:.4f}")

        dt_tr, dt_te, dt_proba = run_baseline_dt(X_train_s, y_train, X_test_s)
        results["decision_tree"] = compute_metrics(dt_tr, dt_te, dt_proba, y_train, y_test)
        logger.info(f"  DT: acc={results['decision_tree']['test_accuracy']:.4f}")

        lr_tr, lr_te, lr_proba = run_baseline_lr(X_train_s, y_train, X_test_s)
        results["logistic_regression"] = compute_metrics(lr_tr, lr_te, lr_proba, y_train, y_test)
        logger.info(f"  LR: acc={results['logistic_regression']['test_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Sklearn baselines failed: {e}")
        logger.debug(traceback.format_exc())
        raise

    # 4. Dictionary stability (5-fold CV) — only if enough training data
    if n_train >= 25:
        logger.info("Running 5-fold CV stability analysis...")
        stability_results = {}
        for K in [3, 5]:
            fold_dicts = []
            fold_accs = []
            try:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
                for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_s, y_train)):
                    ens_f, dict_f = dots_full(
                        X_train_s[tr_idx], y_train[tr_idx], K=K,
                        max_splits=max(5, dots_max_splits // 2),
                        max_trees=3, n_alternation_rounds=2,
                    )
                    fold_dicts.append(dict_f)
                    acc = float(np.mean(ens_f.predict(X_train_s[val_idx]) == y_train[val_idx]))
                    fold_accs.append(acc)
                    logger.info(f"  Stability K={K} fold {fold_idx}: acc={acc:.4f}")
                stab = compute_dictionary_stability(fold_dicts)
                stability_results[f"K{K}"] = {
                    "fold_accuracies": fold_accs,
                    "mean_accuracy": float(np.mean(fold_accs)),
                    "std_accuracy": float(np.std(fold_accs)),
                    "mean_cosine_similarity": stab["mean_cosine"],
                    "pairwise_similarities": stab["pairwise"],
                }
            except Exception as e:
                logger.warning(f"Stability K={K} failed: {e}")
                stability_results[f"K{K}"] = {
                    "fold_accuracies": fold_accs,
                    "mean_accuracy": float(np.mean(fold_accs)) if fold_accs else 0.0,
                    "std_accuracy": 0.0,
                    "mean_cosine_similarity": 0.0,
                    "pairwise_similarities": [],
                }
        results["stability_analysis"] = stability_results
    else:
        logger.info("Skipping stability analysis (n_train < 25)")
        results["stability_analysis"] = {}

    elapsed = time.time() - t0
    logger.info(f"Total experiment time: {elapsed:.1f}s")

    return results, raw_examples, train_indices, test_indices, feature_names


# ===========================================================================
# Section 8: Output Generation
# ===========================================================================
def serialize_results_summary(results):
    summary = {
        "method_accuracies": {},
        "k_sweep_accuracies": {},
        "stability_analysis": results.get("stability_analysis", {}),
        "pareto_frontier": [],
    }
    for method_name, method_res in results.items():
        if method_name == "stability_analysis":
            continue
        if isinstance(method_res, dict) and "test_accuracy" in method_res:
            summary["method_accuracies"][method_name] = method_res["test_accuracy"]
        if method_name.startswith("dots_K"):
            K = method_res.get("K", 0)
            summary["k_sweep_accuracies"][f"K={K}"] = method_res["test_accuracy"]
            summary["pareto_frontier"].append({
                "K": K,
                "test_accuracy": method_res["test_accuracy"],
                "n_unique_directions": K,
                "n_splits": method_res.get("n_splits", 0),
                "direction_names": method_res.get("direction_names", []),
            })
    return summary


def generate_output(results, raw_examples, train_indices, test_indices):
    """Generate output examples in required schema format.

    Schema: {examples: [{input, output, context, dataset, split,
                         predict_baseline, predict_method, method}]}
    """
    primary = "dots_K5"
    baseline = "figs_axis_aligned"

    # Sanity check: make sure primary and baseline exist
    if primary not in results:
        logger.warning(f"Primary method '{primary}' not found, using figs_axis_aligned")
        primary = "figs_axis_aligned"
    if baseline not in results:
        logger.error(f"Baseline '{baseline}' not found!")
        raise KeyError(f"Baseline method '{baseline}' not in results")

    train_counter = 0
    test_counter = 0
    output_examples = []

    for idx, ex in enumerate(raw_examples):
        # Build context without dots_full_results for all except first
        ctx = {}
        for k, v in ex["context"].items():
            ctx[k] = v

        out = {
            "input": ex["input"],
            "output": ex["output"],
            "context": ctx,
            "dataset": ex["dataset"],
            "split": ex["split"],
            "method": primary,
        }

        try:
            if ex["split"] == "train":
                out["predict_baseline"] = str(
                    results[baseline]["train_predictions"][train_counter]
                )
                out["predict_method"] = str(
                    results[primary]["train_predictions"][train_counter]
                )
                train_counter += 1
            else:
                out["predict_baseline"] = str(
                    results[baseline]["test_predictions"][test_counter]
                )
                out["predict_method"] = str(
                    results[primary]["test_predictions"][test_counter]
                )
                test_counter += 1
        except (IndexError, KeyError) as e:
            logger.error(f"Error generating prediction for example {idx}: {e}")
            out["predict_baseline"] = "0"
            out["predict_method"] = "0"

        # Embed full results summary in first example only
        if len(output_examples) == 0:
            out["context"]["dots_full_results"] = serialize_results_summary(results)

        output_examples.append(out)

    logger.info(f"Generated {len(output_examples)} output examples "
                f"(train_counter={train_counter}, test_counter={test_counter})")
    return output_examples


def write_outputs(output_examples, workspace):
    """Write method_out.json + full/mini/preview variants."""
    ws = Path(workspace)

    # Write the single method_out.json that the formatting skill needs
    method_out_path = ws / "method_out.json"
    method_out_path.write_text(json.dumps({"examples": output_examples}, indent=2))
    logger.info(f"Wrote {len(output_examples)} examples to {method_out_path}")

    # Also write the standard full/mini/preview variants directly
    full_path = ws / "full_method_out.json"
    full_path.write_text(json.dumps({"examples": output_examples}, indent=2))

    mini = output_examples[:3]
    mini_path = ws / "mini_method_out.json"
    mini_path.write_text(json.dumps({"examples": mini}, indent=2))

    # Preview: truncate long strings in context
    preview = []
    for ex in output_examples[:3]:
        pex = dict(ex)
        ctx_copy = {}
        for ck, cv in ex["context"].items():
            if ck == "dots_full_results":
                # Keep as dict but truncate nested values
                ctx_copy[ck] = {"summary": "see full_method_out.json for details"}
            elif isinstance(cv, dict) and len(cv) > 5:
                items = list(cv.items())[:5]
                ctx_copy[ck] = dict(items)
            else:
                ctx_copy[ck] = cv
        pex["context"] = ctx_copy
        preview.append(pex)
    preview_path = ws / "preview_method_out.json"
    preview_path.write_text(json.dumps({"examples": preview}, indent=2))

    logger.info(f"Wrote full ({len(output_examples)}), mini ({len(mini)}), "
                f"preview ({len(preview)}) to {ws}")


def print_summary(results):
    """Print human-readable results summary."""
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    for method, res in results.items():
        if method == "stability_analysis":
            continue
        if isinstance(res, dict) and "test_accuracy" in res:
            auroc = res.get("test_auroc", "N/A")
            if isinstance(auroc, float):
                auroc = f"{auroc:.4f}"
            print(f"  {method:30s}  acc={res['test_accuracy']:.4f}  auroc={auroc}")

    if results.get("stability_analysis"):
        print("\nDICTIONARY STABILITY:")
        for k_key, stab in results["stability_analysis"].items():
            print(
                f"  {k_key}: cosine={stab['mean_cosine_similarity']:.4f}  "
                f"cv_acc={stab['mean_accuracy']:.4f}±{stab['std_accuracy']:.4f}"
            )

    print("\nK-SWEEP PARETO:")
    for K in [2, 3, 4, 5, 6, 8, 10]:
        key = f"dots_K{K}"
        if key in results:
            r = results[key]
            print(f"  K={K:2d}  acc={r['test_accuracy']:.4f}  "
                  f"auroc={r.get('test_auroc', 0):.4f}  "
                  f"splits={r.get('n_splits', '?')}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="DOTS Experiment")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit number of examples loaded")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Override data file path")
    args = parser.parse_args()

    data_filepath = args.data_file or str(DATA_DIR / "full_data_out.json")

    logger.info("=" * 65)
    logger.info("DOTS: Dictionary-Constrained Oblique Tree Sums")
    logger.info(f"Data: {data_filepath}")
    logger.info(f"Max examples: {args.max_examples or 'all'}")
    logger.info("=" * 65)

    try:
        results, raw_examples, train_indices, test_indices, feature_names = (
            run_full_experiment(data_filepath, max_examples=args.max_examples)
        )

        output_examples = generate_output(
            results=results,
            raw_examples=raw_examples,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        write_outputs(output_examples, str(WORKSPACE))
        print_summary(results)

        # Final sanity check
        n_out = len(output_examples)
        logger.info(f"SUCCESS: {n_out} examples written")
        if n_out < 50:
            logger.warning(f"Only {n_out} examples produced (minimum 50 recommended)")

    except Exception as e:
        logger.error(f"EXPERIMENT FAILED: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
