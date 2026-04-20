# PROMPT 3: IoT-Shield — Ensemble, Hyperparameter Tuning & Rigorous Benchmarking

## ROLE & CONTEXT

You are an expert ML engineer finalizing **IoT-Shield** NIDS. Phases 1 and 2 are complete. Now we reach the most advanced phase: **ensemble learning + hyperparameter tuning + rigorous validation**.

**Assumed state after Phase 2**:
- ✅ `sample_weight='balanced'` per chunk
- ✅ Scaler fitted from multiple files
- ✅ Bot token in environment
- ✅ Feature engineering (37 → 49 features)
- ✅ Round-robin file streaming
- ✅ Multi-epoch per chunk
- ✅ Stratified test set

**Phase 3 target**: Accuracy ≥ 87%, macro F1 ≥ 78%, per-class F1 ≥ 50% for ALL classes.

## PROJECT STRUCTURE

```
AI Project/
├── main.py
├── ui_design.py
├── data_loader.py
├── trainer.py              ← Major rewrite
├── detector.py             ← Inference updates
├── bot_handler.py
├── guard.py
├── feature_engineering.py  ← From Phase 2
├── ensemble.py             ← NEW (this phase)
├── hyperparameter_tuner.py ← NEW (this phase)
├── benchmark.py            ← NEW (this phase)
└── models/
```

## PHASE 3 OBJECTIVES

### 🟢 ENHANCEMENT #1: Ensemble Learning

**Rationale**: A single linear model (SGDClassifier) has limits. Three diverse linear models voting together typically gain 2-4% accuracy with minimal size cost (~0.03 MB total — still Edge-friendly).

**Approach**: Combine 3 incremental-learning linear classifiers:
- `SGDClassifier(loss='modified_huber')` — probabilistic, outlier-robust (current main model)
- `PassiveAggressiveClassifier` — aggressive on errors, stable on correct predictions
- `SGDClassifier(loss='log_loss')` — logistic, smooth gradients

All three support `partial_fit`, so incremental learning is preserved.

### 🟢 ENHANCEMENT #2: Hyperparameter Tuning

**Rationale**: Current `alpha=1e-4` is sklearn default. For a 34-class imbalanced problem with 49 features, a tuned `alpha` can gain 1-3% accuracy.

**Approach**: Time-budgeted mini grid search on a small sample (~500K rows, not the full dataset) before final training.

### 🟢 ENHANCEMENT #3: Rigorous Benchmarking

**Rationale**: The current reporting (single accuracy/F1 number) is not sufficient for a dissertation defense. We need:
- Per-class precision/recall/F1 table
- Confusion matrix with class names (not just indices)
- Learning curves that reveal overfitting
- Comparison of the 3 individual models vs the ensemble
- Inference latency benchmark (critical for "real-time" claim)

### 🟢 ENHANCEMENT #4: Model Calibration

**Rationale**: The "3-level decision logic" (p < 0.70, p < 0.95) in `detector.py` assumes `predict_proba` returns well-calibrated probabilities. In practice, SGD probabilities are often poorly calibrated. If p=0.95 actually means "75% true positive rate", the threshold is wrong.

**Approach**: Add `CalibratedClassifierCV` wrapper with `method='isotonic'` or `'sigmoid'` on a held-out calibration set.

---

## IMPLEMENTATION SPEC

### Task 1: Ensemble Module

**File**: Create `ensemble.py`

```python
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
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
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
            "passive_aggressive": PassiveAggressiveClassifier(
                C=0.1,
                max_iter=1,
                tol=None,
                random_state=456,
                n_jobs=-1,
                loss='hinge',
            ),
        }
        # Equal weights initially; can be tuned later on validation set
        weights = {"sgd_mhuber": 1.0, "sgd_log": 1.0, "passive_aggressive": 0.8}
        return cls(models, weights)

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Fit all models on a chunk."""
        for name, model in self.models.items():
            try:
                if sample_weight is not None:
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
```

---

### Task 2: Hyperparameter Tuner

**File**: Create `hyperparameter_tuner.py`

```python
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
```

---

### Task 3: Benchmarking Module

**File**: Create `benchmark.py`

```python
"""
IoT-Shield Benchmarking Module
===============================
Produces comprehensive evaluation artifacts for dissertation and analysis.
"""

import os
import json
import time
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


def evaluate_model_full(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Full evaluation suite.
    Returns a dict with all metrics needed for a thesis-grade report.
    """
    t0 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t0

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec_m = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    # Per-class breakdown
    unique_labels = sorted(set(list(y_test) + list(y_pred)))
    per_class = []
    for i, lbl in enumerate(unique_labels):
        name = class_names[lbl] if lbl < len(class_names) else str(lbl)
        support = int(np.sum(y_test == lbl))
        per_class.append({
            "class_id": int(lbl),
            "class_name": name,
            "precision": float(per_class_precision[i]) if i < len(per_class_precision) else 0.0,
            "recall": float(per_class_recall[i]) if i < len(per_class_recall) else 0.0,
            "f1_score": float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
            "support": support,
        })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # Latency benchmark — single-sample predict time
    n_bench = min(1000, len(X_test))
    X_bench = X_test[:n_bench]
    t0 = time.time()
    for i in range(n_bench):
        _ = model.predict(X_bench[i:i+1])
    single_pred_time_ms = (time.time() - t0) / n_bench * 1000

    return {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "accuracy": float(acc),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [
            class_names[l] if l < len(class_names) else str(l) for l in unique_labels
        ],
        "test_set_size": int(len(X_test)),
        "batch_predict_time_s": float(predict_time),
        "single_predict_latency_ms": float(single_pred_time_ms),
        "samples_per_second": int(len(X_test) / predict_time) if predict_time > 0 else 0,
    }


def compare_models(
    results: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Given a list of evaluation dicts (one per model), produce:
      - benchmark_report.json (raw data)
      - benchmark_report.md  (human-readable markdown table)

    Returns path to the markdown report.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Markdown report
    md_path = os.path.join(output_dir, f"benchmark_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# IoT-Shield Benchmark Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Test set size: {results[0].get('test_set_size', '?'):,}\n\n")

        f.write("## Overall metrics\n\n")
        f.write("| Model | Accuracy | F1 (weighted) | F1 (macro) | Latency (ms/sample) |\n")
        f.write("|-------|---------:|--------------:|-----------:|--------------------:|\n")
        for r in results:
            f.write(
                f"| {r['model_name']} | {r['accuracy']:.4f} | "
                f"{r['f1_weighted']:.4f} | {r['f1_macro']:.4f} | "
                f"{r['single_predict_latency_ms']:.3f} |\n"
            )
        f.write("\n")

        f.write("## Per-class F1 — all models\n\n")
        if results:
            class_names = [c["class_name"] for c in results[0]["per_class"]]
            f.write("| Class | " + " | ".join(r["model_name"] for r in results) + " | Support |\n")
            f.write("|-------|" + "|".join(["--------:"] * len(results)) + "|--------:|\n")
            for i, cls in enumerate(class_names):
                row = [cls]
                for r in results:
                    row.append(f"{r['per_class'][i]['f1_score']:.3f}")
                row.append(str(results[0]["per_class"][i]["support"]))
                f.write("| " + " | ".join(row) + " |\n")

    return md_path
```

---

### Task 4: Update `trainer.py` to Use All New Features

**File**: `trainer.py`

1. **Add imports** at top:
```python
from ensemble import IncrementalEnsemble
from hyperparameter_tuner import tune_sgd_hyperparameters
from benchmark import evaluate_model_full, compare_models
```

2. **Add configuration flags** to `TrainerThread`:
```python
class TrainerThread(QThread):
    EPOCHS_PER_CHUNK = 3
    USE_ENSEMBLE = True
    USE_HYPERPARAMETER_TUNING = True
    HP_TUNING_TIME_BUDGET = 300.0  # 5 minutes
```

3. **Add optional hyperparameter tuning step** between scaler fitting and main training loop:

```python
# ═══ 2.5. HYPERPARAMETER TUNING (optional) ═══
best_sgd_params = {"alpha": 1e-4, "loss": "modified_huber", "penalty": "l2"}

if self.USE_HYPERPARAMETER_TUNING:
    self.log_message.emit("\n" + "─" * 40)
    self.log_message.emit("🔍 2.5-BOSQICH: Hyperparameter tuning")
    self.log_message.emit("─" * 40)

    # Build a small sample for tuning (~100K rows from first 3 files)
    sample_X, sample_y = self._build_tuning_sample(dl, n_files=3, rows_per_file=35000)
    # Split 80/20 for tune/val
    split = int(len(sample_X) * 0.8)
    try:
        best_sgd_params, _tune_results = tune_sgd_hyperparameters(
            X_sample=sample_X[:split], y_sample=sample_y[:split],
            classes=all_classes_encoded,
            X_val=sample_X[split:], y_val=sample_y[split:],
            time_budget_seconds=self.HP_TUNING_TIME_BUDGET,
            log_callback=self._emit_log,
        )
    except Exception as e:
        self.log_message.emit(f"   ⚠️ Tuning xato: {e}, defaults ishlatiladi")
```

4. **Add helper method** to build tuning sample:
```python
def _build_tuning_sample(self, dl, n_files=3, rows_per_file=35000):
    """Build a small in-memory sample for HP tuning."""
    Xs, ys = [], []
    import random
    rng = random.Random(42)
    sampled = rng.sample(dl.file_paths, min(n_files, len(dl.file_paths)))
    for fp in sampled:
        for X_chunk, y_chunk in dl.stream_file_chunks(fp, chunksize=rows_per_file):
            Xs.append(X_chunk)
            ys.append(y_chunk)
            break  # only first chunk
    return np.vstack(Xs), np.concatenate(ys)
```

5. **Build model using best params** — replace the old `SGDClassifier(...)` instantiation:

```python
# ═══ 3. MODEL YARATISH ═══

if self.USE_ENSEMBLE:
    self.log_message.emit(f"🎯 Ensemble mode: 3 ta model birga ishlaydi")
    model = IncrementalEnsemble.default_config()
    # Apply tuned hyperparams to the main SGD component if available
    if "sgd_mhuber" in model.models and best_sgd_params.get("loss") == "modified_huber":
        model.models["sgd_mhuber"].alpha = best_sgd_params["alpha"]
        model.models["sgd_mhuber"].penalty = best_sgd_params["penalty"]
        self.log_message.emit(
            f"   ├── Tuned alpha={best_sgd_params['alpha']:.0e} sgd_mhuber ga qo'llandi"
        )
else:
    model = SGDClassifier(
        loss=best_sgd_params["loss"],
        penalty=best_sgd_params["penalty"],
        alpha=best_sgd_params["alpha"],
        max_iter=1, tol=None, random_state=42, n_jobs=-1,
    )
```

6. **Update evaluation section** — replace simple evaluation with `evaluate_model_full`:

```python
# ═══ 5. BAHOLASH (RIGOROUS) ═══
self.log_message.emit("\n" + "─" * 40)
self.log_message.emit("📊 4-BOSQICH: Rigorous Benchmarking")
self.log_message.emit("─" * 40)

# Ensemble overall evaluation
ensemble_eval = evaluate_model_full(
    model, test_X, test_y, dl.class_names,
    model_name="Ensemble" if self.USE_ENSEMBLE else "SGD"
)

# If ensemble, also evaluate each component separately for comparison
all_benchmarks = [ensemble_eval]
if self.USE_ENSEMBLE and hasattr(model, "models"):
    for name, submodel in model.models.items():
        try:
            sub_eval = evaluate_model_full(
                submodel, test_X, test_y, dl.class_names,
                model_name=name
            )
            all_benchmarks.append(sub_eval)
        except Exception as e:
            self.log_message.emit(f"   ⚠️ {name} baholash xato: {e}")

# Write benchmark report to disk
bench_dir = os.path.join(os.path.dirname(MODELS_DIR), "benchmarks")
bench_md = compare_models(all_benchmarks, bench_dir)
self.log_message.emit(f"\n📄 Benchmark report: {bench_md}")

# Log key metrics
self.log_message.emit(f"\n🏆 FINAL METRICS:")
self.log_message.emit(f"   ├── Accuracy:        {ensemble_eval['accuracy']:.4f}")
self.log_message.emit(f"   ├── F1 (weighted):   {ensemble_eval['f1_weighted']:.4f}")
self.log_message.emit(f"   ├── F1 (macro):      {ensemble_eval['f1_macro']:.4f}")
self.log_message.emit(f"   ├── Latency:         {ensemble_eval['single_predict_latency_ms']:.3f} ms/sample")
self.log_message.emit(f"   └── Throughput:      {ensemble_eval['samples_per_second']:,} samples/sec")
```

7. **Update save logic** to handle ensemble:

In `_save_model`, detect if model is an ensemble and save appropriately:
```python
if hasattr(model, "save") and hasattr(model, "models"):
    # It's an ensemble
    pkl_path = os.path.join(MODELS_DIR, f"iot_shield_ensemble_{timestamp}.pkl")
    model.save(pkl_path)
    # Update metadata
    metadata["model_type"] = "IncrementalEnsemble (3-model soft voting)"
    metadata["ensemble_config"] = model.summary()
else:
    # Single model
    pkl_path = os.path.join(MODELS_DIR, f"iot_shield_sgd_{timestamp}.pkl")
    joblib.dump(model, pkl_path, compress=3)
```

---

### Task 5: Update `detector.py` to Load Ensemble

**File**: `detector.py`

1. **Update `load_model`** to detect ensemble vs single model:

```python
def load_model(models_dir: str) -> tuple:
    paths = find_latest_model(models_dir)

    logger.info("📦 Model yuklanmoqda...")

    # Detect ensemble vs single model
    model_file = os.path.basename(paths["pkl"])
    if "ensemble" in model_file.lower():
        from ensemble import IncrementalEnsemble
        model = IncrementalEnsemble.load(paths["pkl"])
        logger.info(f"   ├── Ensemble model: {model_file}")
        logger.info(f"   ├── Komponentlar: {', '.join(model.models.keys())}")
    else:
        model = joblib.load(paths["pkl"])
        logger.info(f"   ├── Model: {model_file}")

    # ... rest of existing code ...
```

---

### Task 6: Optional — Calibration

**File**: Add to `trainer.py` as an OPTIONAL final step (disabled by default since training time is already high):

```python
class TrainerThread(QThread):
    # ...
    USE_CALIBRATION = False  # Enable only if decision thresholds feel "off"

# Inside run(), after evaluation, BEFORE save:
if self.USE_CALIBRATION:
    from sklearn.calibration import CalibratedClassifierCV

    self.log_message.emit("\n📐 Model calibration (isotonic)...")
    # Use last 10% of test set for calibration (don't leak into metrics)
    calib_split = int(len(test_X) * 0.9)
    cal_X, cal_y = test_X[calib_split:], test_y[calib_split:]
    test_X, test_y = test_X[:calib_split], test_y[:calib_split]

    # NOTE: CalibratedClassifierCV requires a fitted base estimator and
    # re-fits on cv='prefit'
    try:
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(cal_X, cal_y)
        model = calibrated
        self.log_message.emit("   └── ✅ Calibration yakunlandi")
    except Exception as e:
        self.log_message.emit(f"   ⚠️ Calibration xato: {e}")
```

---

## TESTING & VALIDATION

1. **Ensemble smoke test**:
   - Create a toy dataset with 3 classes
   - Fit the default ensemble on 10K samples
   - Verify `predict_proba` returns shape `(N, 3)` with rows summing to 1.0 (±0.01)
   - Verify `predict` returns class labels, not indices

2. **Hyperparameter tuner test**:
   - On a 50K sample, run tuner with `time_budget_seconds=60`
   - Verify it returns a valid param dict even if time runs out
   - Verify at least one combination was evaluated

3. **Benchmark output test**:
   - Run `evaluate_model_full` on a toy trained model
   - Verify returned dict has all expected keys
   - Verify `compare_models` produces valid markdown with per-class rows

4. **End-to-end integration test**:
   - Train on 3-file subset with `USE_ENSEMBLE=True` and `USE_HYPERPARAMETER_TUNING=True`
   - Verify the full pipeline completes
   - Verify benchmark report is written
   - Verify ensemble can be saved AND loaded
   - Verify `detector.py` loads the ensemble file and runs inference

5. **Write** `docs/PHASE3_CHANGES.md` with:
   - Summary of all three new modules
   - Benchmark report interpretation guide
   - Expected accuracy/latency trade-offs
   - Dissertation defense talking points for "why ensemble?"

---

## CONSTRAINTS

- Ensemble inference latency should stay under **2× single-model latency**. Benchmark this explicitly.
- All three models must support `partial_fit` (preserves incremental learning).
- Do NOT introduce any non-linear models (no RandomForest, no XGBoost, no neural nets) — this breaks the Edge deployment story of the dissertation.
- Ensemble `.pkl` file size should stay under **5 MB** (vs ~0.01 MB for single SGD). If larger, investigate.
- All existing interfaces (GUI log messages, metadata format, bot alerts) must still work.
- Calibration is OPT-IN — don't enable by default.

## SUCCESS CRITERIA

✅ `ensemble.py`, `hyperparameter_tuner.py`, `benchmark.py` exist and are covered by tests
✅ `IncrementalEnsemble.default_config()` produces a working 3-model ensemble
✅ Hyperparameter tuner picks a param combo within time budget and reports all results
✅ Benchmark report (markdown + JSON) is written after training
✅ Per-class F1 table is part of the report
✅ Inference latency is measured and logged
✅ `detector.py` can load and run the ensemble
✅ `PHASE3_CHANGES.md` documents the defense talking points

## EXPECTED IMPACT

| Metric | Phase 2 | Phase 3 (expected) |
|--------|---------|--------------------|
| Accuracy (weighted) | 82-86% | 85-89% |
| F1 (macro) | 70-78% | 76-82% |
| Per-class F1 worst | ~15-20% (rare classes) | ≥ 45% |
| Model size | 0.01 MB | 0.03-0.05 MB |
| Inference latency | <1 ms | <2 ms |
| Training time | baseline × 3.2 | baseline × 4.5 |

---

**When done**, reply with:
1. Summary of new modules and integration points
2. Benchmark report from a small-scale end-to-end run
3. Dissertation defense bullet points for: "Why ensemble?", "Why these 3 models?", "How does it stay Edge-friendly?"
4. Full content of `PHASE3_CHANGES.md`
5. Any issues discovered and how they were resolved
