"""
IoT-Shield Hyperparameter Tuner
================================
Quick mini-grid-search over a small sample BEFORE running full training.

This is NOT a full cross-validation — it's a practical time-bounded search
to pick better-than-default hyperparameters.
"""

import time
import itertools
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight


def tune_sgd_hyperparameters(
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    classes: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    time_budget_seconds: float = 300.0,
    log_callback=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Grid-search over a small hyperparameter space on a sample.

    Args:
        X_sample, y_sample: training subset (~100K-500K rows)
        classes: all class labels (encoded)
        X_val, y_val: held-out validation subset (~50K rows)
        time_budget_seconds: max time to spend on tuning

    Returns:
        (best_params: dict, all_results: list of dicts)
    """

    # Reasonable ranges for SGDClassifier on imbalanced multi-class
    param_grid = {
        "alpha": [1e-5, 1e-4, 5e-4, 1e-3],
        "loss": ["modified_huber", "log_loss"],
        "penalty": ["l2", "elasticnet"],
    }

    # Compute sample weights once
    sample_weights = compute_sample_weight("balanced", y=y_sample)

    combinations = list(itertools.product(
        param_grid["alpha"],
        param_grid["loss"],
        param_grid["penalty"],
    ))

    if log_callback:
        log_callback(f"🔍 Hyperparameter tuning: {len(combinations)} kombinatsiya")
        log_callback(f"   ├── Vaqt limiti: {time_budget_seconds:.0f} soniya")
        log_callback(f"   ├── Sample: {len(X_sample):,} qator")
        log_callback(f"   └── Validation: {len(X_val):,} qator")

    all_results = []
    best_f1 = -1
    best_params = None
    start_time = time.time()

    for i, (alpha, loss, penalty) in enumerate(combinations):
        elapsed = time.time() - start_time
        if elapsed > time_budget_seconds:
            if log_callback:
                log_callback(f"   ⏱️ Vaqt tugadi, {i}/{len(combinations)} kombinatsiya sinalgan")
            break

        try:
            l1_ratio = 0.15 if penalty == "elasticnet" else None
            kwargs = {
                "loss": loss,
                "penalty": penalty,
                "alpha": alpha,
                "max_iter": 5,  # A few more iterations on sample
                "tol": 1e-3,
                "random_state": 42,
                "n_jobs": -1,
            }
            if l1_ratio is not None:
                kwargs["l1_ratio"] = l1_ratio

            model = SGDClassifier(**kwargs)

            # Full fit on sample (not partial_fit — grid search, not streaming)
            model.fit(X_sample, y_sample, sample_weight=sample_weights)

            y_pred = model.predict(X_val)
            f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

            result = {
                "alpha": alpha,
                "loss": loss,
                "penalty": penalty,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "time_s": time.time() - elapsed - start_time,
            }
            all_results.append(result)

            # Optimize for macro F1 (imbalanced-dataset-appropriate)
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_params = {
                    "alpha": alpha,
                    "loss": loss,
                    "penalty": penalty,
                }

            if log_callback:
                log_callback(
                    f"   ├── [{i+1}/{len(combinations)}] "
                    f"alpha={alpha:.0e} loss={loss[:8]:<8s} penalty={penalty[:7]:<7s} "
                    f"→ F1_macro={f1_macro:.4f} F1_w={f1_weighted:.4f}"
                )

        except Exception as e:
            if log_callback:
                log_callback(f"   ⚠️ Kombinatsiya xato: {str(e)[:80]}")
            continue

    if best_params is None:
        if log_callback:
            log_callback("   ⚠️ Hech qanday kombinatsiya muvaffaqiyatli o'tmadi. Default qaytariladi.")
        best_params = {"alpha": 1e-4, "loss": "modified_huber", "penalty": "l2"}

    if log_callback:
        log_callback(
            f"\n🏆 Eng yaxshi: alpha={best_params['alpha']:.0e} "
            f"loss={best_params['loss']} penalty={best_params['penalty']} "
            f"(F1_macro={best_f1:.4f})"
        )

    return best_params, all_results
