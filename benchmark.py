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
