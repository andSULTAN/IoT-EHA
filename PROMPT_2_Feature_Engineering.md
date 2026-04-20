# PROMPT 2: IoT-Shield — Feature Engineering & Training Loop Improvements

## ROLE & CONTEXT

You are an expert ML engineer continuing work on **IoT-Shield**, a real-time NIDS for IoT networks. **Phase 1 (critical fixes + security) is COMPLETE**. Now we move to Phase 2: improving data quality and training dynamics.

**Assumed state after Phase 1**:
- `sample_weight='balanced'` is used per chunk ✅
- Scaler is fitted from multiple random files ✅
- Bot token is in environment variable ✅
- Macro-averaged metrics are tracked ✅

**Phase 2 target**: Accuracy ≥ 83%, macro F1 ≥ 72%.

## PROJECT STRUCTURE

```
AI Project/
├── main.py
├── ui_design.py
├── data_loader.py     ← Main changes here
├── trainer.py         ← Main changes here
├── detector.py        ← Small changes (feature parity)
├── bot_handler.py
├── guard.py
└── models/
```

## PHASE 2 OBJECTIVES

### 🟡 ISSUE #4: SGD Only Sees Each Chunk Once (max_iter=1)

**Problem**: In `trainer.py`, `SGDClassifier(max_iter=1, tol=None)` means each chunk is fed to the model once. For stochastic gradient descent to converge well, each data point ideally contributes to multiple weight updates.

**Fix**: Loop over each chunk N times (`EPOCHS_PER_CHUNK = 3`), with reshuffling between epochs.

### 🟡 ISSUE #5: No Feature Engineering

**Problem**: The raw 37 CICIOT2023 features are fed directly into the model. Derived features (ratios, ranges, intensity metrics) often carry more discriminative signal than raw features.

**Fix**: Add ~10-13 derived features → total 47-50 features. Must be computed **identically** in both `data_loader.py` (training) and `detector.py` (inference).

### 🟡 ISSUE #6: Chunks Delivered in File Order (Catastrophic Forgetting)

**Problem**: In `data_loader.py`, `stream_file_chunks` reads one file completely before moving to the next. Since CICIOT2023 files are grouped by attack class, the model sees e.g. 100% DDoS for a long stretch, then 100% BENIGN, etc. This causes *catastrophic forgetting* — the model overfits to the most recently seen class.

**Fix**: Add a "round-robin" streaming mode that interleaves chunks from all files.

### 🟡 ISSUE #7: Validation Set Is Biased

**Problem**: The test set is carved out of the *last file's first chunk* (20%). If the last file is a single-class file (very likely in CICIOT2023), the test set is homogeneous and accuracy is inflated.

**Fix**: Build a stratified test set from small samples across ALL files.

---

## IMPLEMENTATION SPEC

### Task 1: Feature Engineering Module

**File**: Create a new file `feature_engineering.py` in the project root.

```python
"""
IoT-Shield Feature Engineering Module
======================================
Derives additional features from the raw 37 CICIOT2023 features.

CRITICAL: This module MUST be used identically in:
  - data_loader.py (during training)
  - detector.py   (during inference)

Changing this file requires retraining the model.
"""

import numpy as np
import pandas as pd
from typing import List

# ═══════════════════════════════════════════════════════════
# Feature engineering version — bump when formulas change
# ═══════════════════════════════════════════════════════════
FEATURE_ENG_VERSION = "1.0.0"

# Small epsilon to avoid division by zero
EPS = 1e-6


# List of derived feature names, in the exact order they'll be appended.
# This is THE canonical order — detector.py must use this same order.
DERIVED_FEATURES: List[str] = [
    "syn_to_fin_ratio",        # SYN flood signature
    "ack_to_syn_ratio",        # Normal handshake balance
    "rst_ratio",               # Connection reset rate
    "size_variance_ratio",     # Packet size irregularity
    "size_range",              # Max - Min packet size
    "pkts_per_second",         # Rate intensity
    "bytes_per_packet",        # Average payload ratio
    "flag_diversity",          # How many flag types seen
    "has_web_traffic",         # HTTP or HTTPS present
    "has_system_traffic",      # SSH, Telnet, DNS present
    "protocol_mix",            # Number of distinct protocols
    "high_frequency_flag",     # Rate > threshold indicator
]


def add_derived_features_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas DataFrame version — used in data_loader.py during training.
    Accepts a DataFrame with the 37 raw CICIOT2023 columns.
    Returns a DataFrame with 37 + len(DERIVED_FEATURES) columns.
    """
    X = X.copy()

    # Rate-based ratios (flag activity)
    X["syn_to_fin_ratio"] = X["syn_count"] / (X["fin_count"] + EPS)
    X["ack_to_syn_ratio"] = X["ack_count"] / (X["syn_count"] + EPS)
    X["rst_ratio"] = X["rst_count"] / (X["Number"] + EPS)

    # Packet size features
    X["size_variance_ratio"] = X["Variance"] / (X["AVG"] + EPS)
    X["size_range"] = X["Max"] - X["Min"]
    X["bytes_per_packet"] = X["Tot size"] / (X["Number"] + EPS)

    # Flow intensity
    X["pkts_per_second"] = X["Rate"]  # Rate is already pkts/sec in CICIOT2023

    # Protocol mix
    protocol_cols = ["HTTP", "HTTPS", "DNS", "SSH", "Telnet", "SMTP", "IRC",
                     "TCP", "UDP", "DHCP", "ARP", "ICMP"]
    available_protos = [c for c in protocol_cols if c in X.columns]
    if available_protos:
        X["protocol_mix"] = X[available_protos].sum(axis=1)
    else:
        X["protocol_mix"] = 0.0

    # Flag diversity (how many distinct flag types are set)
    flag_cols = ["fin_flag_number", "syn_flag_number", "rst_flag_number",
                 "psh_flag_number", "ack_flag_number", "ece_flag_number",
                 "cwr_flag_number"]
    available_flags = [c for c in flag_cols if c in X.columns]
    if available_flags:
        X["flag_diversity"] = X[available_flags].sum(axis=1)
    else:
        X["flag_diversity"] = 0.0

    # Web / system traffic flags
    X["has_web_traffic"] = ((X.get("HTTP", 0) > 0) | (X.get("HTTPS", 0) > 0)).astype(np.float32)
    X["has_system_traffic"] = (
        (X.get("SSH", 0) > 0)
        | (X.get("Telnet", 0) > 0)
        | (X.get("DNS", 0) > 0)
    ).astype(np.float32)

    # High-frequency indicator (Rate > 1000 pkts/sec → likely flood)
    X["high_frequency_flag"] = (X["Rate"] > 1000.0).astype(np.float32)

    # Clean up infinities / NaNs produced by ratios
    derived_col_names = DERIVED_FEATURES
    for col in derived_col_names:
        if col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            X[col] = X[col].astype(np.float32)

    return X


def add_derived_features_np(features: np.ndarray, feature_order: List[str]) -> np.ndarray:
    """
    NumPy version — used in detector.py during inference.
    Accepts a 1D array of 37 raw feature values (in feature_order).
    Returns a 1D array of 37 + len(DERIVED_FEATURES) values.

    feature_order: the order of the raw features in `features` array.
    """
    # Build lookup dict
    fmap = {name: float(features[i]) for i, name in enumerate(feature_order)}

    def g(name: str, default: float = 0.0) -> float:
        return fmap.get(name, default)

    derived = [
        g("syn_count") / (g("fin_count") + EPS),                              # syn_to_fin_ratio
        g("ack_count") / (g("syn_count") + EPS),                              # ack_to_syn_ratio
        g("rst_count") / (g("Number") + EPS),                                 # rst_ratio
        g("Variance") / (g("AVG") + EPS),                                     # size_variance_ratio
        g("Max") - g("Min"),                                                  # size_range
        g("Rate"),                                                            # pkts_per_second
        g("Tot size") / (g("Number") + EPS),                                  # bytes_per_packet
        (g("fin_flag_number") + g("syn_flag_number") + g("rst_flag_number")
         + g("psh_flag_number") + g("ack_flag_number")
         + g("ece_flag_number") + g("cwr_flag_number")),                      # flag_diversity
        1.0 if (g("HTTP") > 0 or g("HTTPS") > 0) else 0.0,                    # has_web_traffic
        1.0 if (g("SSH") > 0 or g("Telnet") > 0 or g("DNS") > 0) else 0.0,    # has_system_traffic
        (g("HTTP") + g("HTTPS") + g("DNS") + g("SSH") + g("Telnet")
         + g("SMTP") + g("IRC") + g("TCP") + g("UDP") + g("DHCP")
         + g("ARP") + g("ICMP")),                                             # protocol_mix
        1.0 if g("Rate") > 1000.0 else 0.0,                                   # high_frequency_flag
    ]

    # Clean infinities / NaNs
    derived_clean = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in derived]

    return np.concatenate([features, np.array(derived_clean, dtype=np.float32)])
```

### Task 2: Integrate Feature Engineering in `data_loader.py`

**File**: `data_loader.py`

1. **Add import** at the top:
```python
from feature_engineering import add_derived_features_df, DERIVED_FEATURES, FEATURE_ENG_VERSION
```

2. **Update `stream_file_chunks`** method — after the line `X = X.replace([np.inf, -np.inf], np.nan).fillna(0)` and BEFORE `X = downcast_dataframe(X)`, add:

```python
# === FEATURE ENGINEERING ===
# Add derived features (version: FEATURE_ENG_VERSION)
X = add_derived_features_df(X)
```

3. **Update `fit_scaler_from_samples`** — apply feature engineering BEFORE fitting the scaler. After the line `feature_data = feature_data.replace([np.inf, -np.inf], np.nan).fillna(0)`, add:

```python
# Apply feature engineering BEFORE scaler fitting
# (scaler must learn stats of derived features too)
feature_data = add_derived_features_df(feature_data)
```

4. **Update `feature_columns` property** — it should now include both raw and derived features.

Add a new property method to the `DataLoader` class:

```python
def get_all_feature_names(self) -> List[str]:
    """Return raw + derived feature names in the exact training order."""
    return list(self.feature_columns) + list(DERIVED_FEATURES)
```

5. **Update `get_summary`** to include feature engineering version:
```python
def get_summary(self) -> dict:
    ...
    return {
        ...
        "feature_count": len(self.feature_columns),
        "derived_feature_count": len(DERIVED_FEATURES),
        "total_feature_count": len(self.feature_columns) + len(DERIVED_FEATURES),
        "feature_eng_version": FEATURE_ENG_VERSION,
        ...
    }
```

---

### Task 3: Round-Robin File Streaming

**File**: `data_loader.py`

**Add a new method** to `DataLoader` class (doesn't replace existing `stream_file_chunks`):

```python
def stream_all_files_round_robin(
    self,
    chunksize: int = 100000,
    shuffle_chunks: bool = True,
    random_state: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Interleave chunks from ALL files — prevents catastrophic forgetting.

    Instead of reading file_1 completely, then file_2, this reads one chunk
    from each file in rotation. This keeps the class distribution balanced
    across time, which is critical for SGD convergence.

    Example with 3 files:
        Order: file1_chunk0, file2_chunk0, file3_chunk0,
               file1_chunk1, file2_chunk1, file3_chunk1, ...
    """
    from sklearn.utils import shuffle as sklearn_shuffle

    # Create generators for each file
    file_gens = []
    for fp in self.file_paths:
        file_gens.append(self.stream_file_chunks(fp, chunksize=chunksize))

    # Round-robin until all generators are exhausted
    active_gens = list(file_gens)
    rotation = 0
    while active_gens:
        still_active = []
        for gen in active_gens:
            try:
                X_chunk, y_chunk = next(gen)

                if shuffle_chunks and len(X_chunk) > 0:
                    X_chunk, y_chunk = sklearn_shuffle(
                        X_chunk, y_chunk,
                        random_state=random_state + rotation
                    )

                yield X_chunk, y_chunk
                still_active.append(gen)

            except StopIteration:
                continue

        active_gens = still_active
        rotation += 1
```

---

### Task 4: Stratified Test Set Builder

**File**: `data_loader.py`

**Add a new method**:

```python
def build_stratified_test_set(
    self,
    rows_per_file: int = 2000,
    max_rows_per_class: int = 5000,
    random_state: int = 42,
    log_callback=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a test set that samples across ALL files AND balances classes.

    Strategy:
      1. Read a small sample from each file.
      2. Concatenate and apply feature engineering + scaling.
      3. Limit samples per class to max_rows_per_class.
      4. Return (X_test_scaled, y_test_encoded).
    """
    if self.scaler is None or self.label_encoder is None:
        raise DataValidationError("Avval scaler va encoder ni fit qiling!")

    if log_callback:
        log_callback(f"🎯 Stratified test set yaratilmoqda...")

    all_X = []
    all_y = []
    samples_per_class = defaultdict(int)

    for idx, fp in enumerate(self.file_paths):
        try:
            df = pd.read_csv(fp, nrows=rows_per_file, low_memory=False)
            df.columns = df.columns.str.strip()

            if self.label_column not in df.columns:
                continue

            # Filter known labels
            raw_labels = df[self.label_column].astype(str)
            known_mask = raw_labels.isin(self.class_names)
            df = df[known_mask]

            if len(df) == 0:
                continue

            # Balance per class
            kept_rows = []
            for cls in df[self.label_column].unique():
                if samples_per_class[cls] >= max_rows_per_class:
                    continue
                cls_rows = df[df[self.label_column] == cls]
                remaining = max_rows_per_class - samples_per_class[cls]
                to_take = min(len(cls_rows), remaining)
                kept_rows.append(cls_rows.head(to_take))
                samples_per_class[cls] += to_take

            if not kept_rows:
                continue

            df_kept = pd.concat(kept_rows, ignore_index=True)

            X = df_kept[self.feature_columns].copy()
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            X = add_derived_features_df(X)
            X = downcast_dataframe(X)

            y = self.label_encoder.transform(
                df_kept[self.label_column].astype(str)
            )

            X_scaled = self.scaler.transform(X.values).astype(np.float32)

            all_X.append(X_scaled)
            all_y.append(y)

        except Exception as e:
            if log_callback:
                log_callback(f"   ⚠️ {os.path.basename(fp)}: {str(e)[:100]}")
            continue

    if not all_X:
        raise DataValidationError("Test set uchun ma'lumot topilmadi!")

    X_test = np.vstack(all_X)
    y_test = np.concatenate(all_y)

    if log_callback:
        log_callback(f"   ├── Test set hajmi: {len(X_test):,} qator")
        log_callback(f"   ├── Sinflar: {len(samples_per_class)} ta")
        log_callback(f"   └── ✅ Stratified test set tayyor")

    return X_test, y_test
```

---

### Task 5: Update `trainer.py` — Use New Methods + Multi-Epoch

**File**: `trainer.py`

1. **Add configurable epochs** as a class attribute:
```python
class TrainerThread(QThread):
    # ...
    EPOCHS_PER_CHUNK = 3  # Each chunk is revisited 3 times
```

2. **Rewrite the training loop** — replace the existing per-file, per-chunk loops with round-robin streaming and per-chunk epochs:

**BEFORE** (current code around line 120-170):
```python
for file_idx, file_path in enumerate(file_paths):
    # ... per-file loop ...
    for X_chunk, y_chunk in dl.stream_file_chunks(file_path, chunksize=100000):
        # ... training ...
        model.partial_fit(X_chunk, y_chunk, classes=all_classes_encoded)
```

**AFTER**:
```python
self.log_message.emit(f"🔄 Round-robin streaming: {total_files} ta fayldan chunks interleave qilinadi")
self.log_message.emit(f"📚 Har chunk {self.EPOCHS_PER_CHUNK} marta ko'riladi (epochs)")
self.log_message.emit("")

total_chunks_processed = 0
# Estimate total chunks for progress (rough)
total_rows_estimate = dl.total_rows or (total_files * 500000)
estimated_chunks = total_rows_estimate // 100000

for X_chunk, y_chunk in dl.stream_all_files_round_robin(
    chunksize=100000,
    shuffle_chunks=True,
    random_state=42,
):
    if self._is_cancelled:
        return self._cancel()

    if len(X_chunk) == 0:
        continue

    try:
        # Compute sample weights (class imbalance)
        sample_weights = compute_sample_weight('balanced', y=y_chunk)

        # Multi-epoch per chunk
        for epoch in range(self.EPOCHS_PER_CHUNK):
            if epoch > 0:
                # Reshuffle between epochs
                X_chunk, y_chunk, sample_weights = sklearn_shuffle(
                    X_chunk, y_chunk, sample_weights,
                    random_state=42 + total_chunks_processed * 10 + epoch
                )

            model.partial_fit(
                X_chunk, y_chunk,
                classes=all_classes_encoded,
                sample_weight=sample_weights,
            )

        total_samples_trained += len(X_chunk)
        total_chunks_processed += 1

        # Periodic logging
        if total_chunks_processed % 10 == 0:
            self.log_message.emit(
                f"   ⚙️ Chunks: {total_chunks_processed}, "
                f"Qatorlar: {total_samples_trained:,}"
            )

        # Progress bar update
        if estimated_chunks > 0:
            progress = training_progress_base + int(
                total_chunks_processed / estimated_chunks * training_progress_range
            )
            self.progress_updated.emit(min(progress, 90))

    except Exception as e:
        self.log_message.emit(f"   ⚠️ Chunk xato: {str(e)[:120]}")
        continue

    # Memory cleanup every 20 chunks
    if total_chunks_processed % 20 == 0:
        gc.collect()

self.log_message.emit(f"\n✅ Jami {total_samples_trained:,} qator ({total_chunks_processed} chunk) o'qitildi")
```

3. **Replace the single-file test set extraction** with stratified test set. Before this loop, REMOVE the code that extracts test_X/test_y from the last file's first chunk.

AFTER the training loop, add:
```python
# === Build stratified test set ===
self.log_message.emit("\n" + "─" * 40)
self.log_message.emit("🎯 Stratified test set yaratilmoqda...")
self.log_message.emit("─" * 40)

test_X, test_y = dl.build_stratified_test_set(
    rows_per_file=2000,
    max_rows_per_class=3000,
    log_callback=self._emit_log,
)
```

4. **Update metadata saving** (`_save_model` method) to include feature engineering info:

```python
metadata = {
    ...
    "feature_eng_version": FEATURE_ENG_VERSION,
    "raw_feature_names": dl.feature_columns,
    "derived_feature_names": list(DERIVED_FEATURES),
    "all_feature_names": dl.get_all_feature_names(),
    "total_features": len(dl.feature_columns) + len(DERIVED_FEATURES),
    "epochs_per_chunk": self.EPOCHS_PER_CHUNK,
    ...
}
```

Add the import at the top of `trainer.py`:
```python
from feature_engineering import DERIVED_FEATURES, FEATURE_ENG_VERSION
```

---

### Task 6: Update `detector.py` to Use Same Features

**File**: `detector.py`

1. **Add import** at top:
```python
from feature_engineering import add_derived_features_np, DERIVED_FEATURES, FEATURE_ENG_VERSION
```

2. **Update `FEATURE_ORDER`** to keep the raw 37 features as-is (don't append derived features here — they're computed on-the-fly).

3. **In `_predict_and_act` method**, modify the feature pipeline. Find this section:

```python
X = features.reshape(1, -1)
X_scaled = self.scaler.transform(X).astype(np.float32)
```

Replace with:

```python
# Apply feature engineering (must match training — same version!)
features_extended = add_derived_features_np(features, FEATURE_ORDER)
X = features_extended.reshape(1, -1)
X_scaled = self.scaler.transform(X).astype(np.float32)
```

4. **Add version compatibility check** when loading the model. In `load_model` function:

```python
def load_model(models_dir: str) -> tuple:
    paths = find_latest_model(models_dir)
    # ... existing code ...

    meta = paths["metadata"]

    # Feature engineering version check
    model_fe_version = meta.get("feature_eng_version", "unknown")
    if model_fe_version != FEATURE_ENG_VERSION:
        logger.warning(
            f"⚠️ Feature engineering version mismatch!\n"
            f"   Model trained with: {model_fe_version}\n"
            f"   Current code: {FEATURE_ENG_VERSION}\n"
            f"   Please retrain the model!"
        )

    # ... rest of existing code ...
```

---

## TESTING

1. **Unit test `feature_engineering.py`**:
   - Create a test with a sample DataFrame (few rows, 37 raw features)
   - Verify `add_derived_features_df` returns DataFrame with 37 + 12 = 49 columns
   - Verify no NaN/Inf in output
   - Verify the numpy version produces identical values as the pandas version on the same input

2. **Integration test for round-robin**:
   - Load 3 small test files (different classes)
   - Iterate through `stream_all_files_round_robin` and record the order of files chunks come from
   - Verify the order is interleaved (not file1, file1, file1, then file2, file2, file2)

3. **Stratified test set test**:
   - Build a test set from 5 files
   - Verify each class has roughly similar count (bounded by `max_rows_per_class`)
   - Verify no single class dominates

4. **End-to-end smoke test**:
   - Train on a small subset (3 files, 100K rows each) using new pipeline
   - Verify training completes without errors
   - Verify saved model has all new metadata fields

5. **Write summary** in `docs/PHASE2_CHANGES.md`.

---

## CONSTRAINTS

- **Feature engineering formulas must be identical** between `data_loader.py` (training) and `detector.py` (inference). This is enforced by using the shared `feature_engineering.py` module.
- `FEATURE_ENG_VERSION` must be bumped if formulas change. Models trained with a different version will log a warning at inference time.
- Existing `.pkl` models from Phase 1 will NOT be compatible with the new inference code (since they don't have derived features). Document this clearly in `PHASE2_CHANGES.md`.
- Memory: round-robin with 63 open file handles may exceed OS limits. If so, fall back to processing files in batches of 10.
- Keep backward compat: `stream_file_chunks` (old method) should still work, just not used by the new trainer.

## SUCCESS CRITERIA

✅ `feature_engineering.py` module exists, imported by both training and inference
✅ Training loop uses round-robin streaming + per-chunk epochs (3×)
✅ Stratified test set is built from all files, classes are balanced
✅ Model metadata includes `feature_eng_version` and new feature lists
✅ `detector.py` inference produces identical feature vectors as training pipeline
✅ A full training run on a small subset completes successfully
✅ `PHASE2_CHANGES.md` documents all changes

## EXPECTED IMPACT

| Metric | Phase 1 | Phase 2 (expected) |
|--------|---------|--------------------|
| Accuracy (weighted) | 78-82% | 82-86% |
| F1 (macro) | 60-70% | 70-78% |
| Training time | baseline × 1.05 | baseline × 3.2 (due to 3× epochs) |
| Features | 37 | 49 |

Training time tripling is expected — the 3-epoch loop is the main cost. If unacceptable, reduce to `EPOCHS_PER_CHUNK = 2` (still better than 1).

---

**When done**, reply with:
1. Diff summary per file
2. Confirmation that training pipeline runs end-to-end
3. Key metrics from a small-scale test run
4. Full content of `PHASE2_CHANGES.md`
