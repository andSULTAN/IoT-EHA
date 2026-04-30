"""
IoT-Shield Ensemble Module
===========================
Combines multiple incremental-learning linear classifiers via
soft voting (averaging predict_proba outputs).

All component models support partial_fit, so the full incremental
learning pipeline is preserved.
"""

import numpy as np
import joblib
from typing import List, Dict, Tuple, Any, Optional
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight


class IncrementalEnsemble:
    """
    Soft-voting ensemble over multiple incremental linear classifiers.

    Usage:
        ens = IncrementalEnsemble.default_config()
        for X_chunk, y_chunk in stream:
            ens.partial_fit(X_chunk, y_chunk, classes=all_classes)
        probs = ens.predict_proba(X_test)
        preds = ens.predict(X_test)
    """

    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        Args:
            models: dict of name -> sklearn classifier (supporting partial_fit)
            weights: optional dict of name -> float (for weighted voting)
                     If None, uses uniform weights.
        """
        self.models = models
        self.weights = weights or {name: 1.0 for name in models}
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive value")
        self.weights = {k: v / total for k, v in self.weights.items()}

    @classmethod
    def default_config(cls) -> "IncrementalEnsemble":
        """Create the standard 3-model ensemble."""
        models = {
            "sgd_mhuber": SGDClassifier(
                loss='modified_huber',
                penalty='l2',
                alpha=1e-4,
                max_iter=1,
                tol=None,
                random_state=42,
                n_jobs=-1,
            ),
            "sgd_log": SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=1e-4,
                max_iter=1,
                tol=None,
                random_state=123,
                n_jobs=-1,
            ),
            "sgd_pa": SGDClassifier(
                loss='modified_huber',
                penalty='l1',
                alpha=5e-5,
                max_iter=1,
                tol=None,
                random_state=456,
                n_jobs=-1,
            ),
        }
        weights = {"sgd_mhuber": 1.0, "sgd_log": 1.0, "sgd_pa": 0.8}
        return cls(models, weights)

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Fit all models on a chunk."""
        import inspect
        for name, model in self.models.items():
            try:
                # Check if partial_fit accepts sample_weight
                sig = inspect.signature(model.partial_fit)
                supports_sample_weight = 'sample_weight' in sig.parameters

                if sample_weight is not None and supports_sample_weight:
                    model.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
                else:
                    model.partial_fit(X, y, classes=classes)
            except Exception as e:
                # One model failing shouldn't kill the whole ensemble
                import logging
                logging.getLogger("Ensemble").warning(
                    f"Model '{name}' partial_fit failed: {e}"
                )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted average of predict_proba across all models.
        Models that don't support predict_proba (e.g. PassiveAggressive)
        use decision_function with softmax conversion.
        """
        all_probs = []
        weight_sum = 0.0

        for name, model in self.models.items():
            w = self.weights[name]
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                else:
                    # Fallback: softmax of decision function
                    scores = model.decision_function(X)
                    if scores.ndim == 1:
                        scores = np.vstack([-scores, scores]).T
                    # Numerically stable softmax
                    scores = scores - scores.max(axis=1, keepdims=True)
                    exp_scores = np.exp(scores)
                    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

                all_probs.append(w * probs)
                weight_sum += w

            except Exception:
                continue

        if not all_probs or weight_sum == 0:
            # Degenerate case — shouldn't happen normally
            n_samples = X.shape[0]
            n_classes = len(list(self.models.values())[0].classes_)
            return np.ones((n_samples, n_classes)) / n_classes

        return np.sum(all_probs, axis=0) / weight_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard prediction based on averaged probabilities."""
        probs = self.predict_proba(X)
        # Use the first model's classes_ as canonical
        classes = list(self.models.values())[0].classes_
        return classes[np.argmax(probs, axis=1)]

    @property
    def classes_(self) -> np.ndarray:
        """Expose classes_ to mimic sklearn classifier interface."""
        return list(self.models.values())[0].classes_

    def save(self, path: str):
        """Save the entire ensemble to one .pkl file."""
        joblib.dump(
            {
                "models": self.models,
                "weights": self.weights,
                "version": "ensemble-1.0",
            },
            path,
            compress=3,
        )

    @classmethod
    def load(cls, path: str) -> "IncrementalEnsemble":
        data = joblib.load(path)
        return cls(models=data["models"], weights=data["weights"])

    def summary(self) -> Dict[str, Any]:
        """Human-readable summary of the ensemble."""
        return {
            "n_models": len(self.models),
            "model_names": list(self.models.keys()),
            "weights": self.weights,
        }
