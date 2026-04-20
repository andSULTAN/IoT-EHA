# PROMPT 1: IoT-Shield — Critical Fixes & Security Hardening

## ROLE & CONTEXT

You are an expert ML engineer and Python developer working on **IoT-Shield**, a real-time Network Intrusion Detection System (NIDS) for IoT networks. The system uses `SGDClassifier` with incremental learning on the CICIOT2023 dataset (45M+ rows, 34 attack classes).

**Current model performance**: Accuracy 74.86%, F1-Score 72.40%.
**Target after this phase**: Accuracy ≥ 80%, macro F1 ≥ 75%.

## PROJECT STRUCTURE

```
AI Project/
├── main.py            # PyQt6 GUI controller
├── ui_design.py       # GUI design
├── data_loader.py     # CICIOT2023 loading + streaming
├── trainer.py         # SGDClassifier training engine
├── detector.py        # Real-time inference engine
├── bot_handler.py     # Telegram bot (aiogram 3.x)
├── guard.py           # Main entry point
└── models/            # Saved .pkl files
```

## PHASE 1 OBJECTIVES (THIS PROMPT)

Fix three critical issues that are currently hurting accuracy and security:

### 🔴 CRITICAL ISSUE #1: Class Imbalance Not Handled

**Problem**: `trainer.py` uses `class_weight=None`. CICIOT2023 has extreme imbalance (DDoS-ICMP_Flood: 7.2M samples vs Uploading_Attack: 1,250 samples — ratio ~5800:1). The model essentially ignores minority classes. The 74.86% accuracy is misleading because weighted metrics hide this.

**Why this matters**: In a NIDS, missing a rare but dangerous attack (e.g., SQL Injection, Brute Force) is worse than missing a common DDoS flood. The current model is biased toward majority classes.

**Fix**: Use `sklearn.utils.class_weight.compute_sample_weight('balanced', y_chunk)` per chunk and pass it to `partial_fit` via the `sample_weight` parameter.

### 🔴 CRITICAL ISSUE #2: Scaler Fitted From Single File (Distribution Bias)

**Problem**: `data_loader.py → fit_scaler_from_first_file()` fits `StandardScaler` using ONLY the first CSV file. CICIOT2023 files are grouped by attack type, so the first file may contain only one attack class (e.g., all DDoS). This gives the scaler a biased view of feature distributions, causing incorrect normalization for other classes during training and inference.

**Fix**: Sample from multiple files (e.g., 5-8 files randomly) and use `partial_fit` progressively. Rename the function to `fit_scaler_from_samples`.

### 🔴 CRITICAL ISSUE #3: Bot Token Exposed in Source Code

**Problem**: `bot_handler.py` line 41 has the Telegram bot token hardcoded:
```python
BOT_TOKEN = "8789775060:AAF8UmdufcsJFCfwJkudz8LV9n5Cz4So_ag"
```

If this code was ever committed to a public git repo, anyone can take over the bot and:
- Spam all subscribers
- Read all chat IDs
- Block legitimate IPs via the bot commands
- Impersonate the admin

**Fix**: Move to environment variable with a `.env` file fallback using `python-dotenv`.

---

## IMPLEMENTATION SPEC

### Task 1: Fix `trainer.py` — Class Imbalance

**File**: `trainer.py`

**Changes inside the training loop** (inside `run()` method, around line 145-170 where `model.partial_fit` is called):

1. Add import at the top:
```python
from sklearn.utils.class_weight import compute_sample_weight
```

2. In the training loop, for each chunk, compute sample weights BEFORE calling `partial_fit`:

```python
# Compute per-sample weights to handle class imbalance
# 'balanced' mode: weight = n_samples / (n_classes * np.bincount(y))
try:
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_chunk
    )
except Exception as e:
    self.log_message.emit(f"   ⚠️ sample_weight hisoblashda xato: {e}")
    sample_weights = None

# Partial fit WITH sample weights
if sample_weights is not None:
    model.partial_fit(
        X_chunk, y_chunk,
        classes=all_classes_encoded,
        sample_weight=sample_weights
    )
else:
    model.partial_fit(X_chunk, y_chunk, classes=all_classes_encoded)
```

3. Add a log line (only once, before the training loop starts) explaining the change:
```python
self.log_message.emit("   ⚖️ Class imbalance handling: sample_weight='balanced' ishlatilmoqda")
```

4. **Add macro-averaged metrics** to `_evaluate_model` (currently only weighted is computed). Add these alongside existing metrics:

```python
# Macro-averaged (treats all classes equally — important for imbalanced data)
precision_macro = precision_score(test_y, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(test_y, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(test_y, y_pred, average='macro', zero_division=0)

self.log_message.emit(f"   ├── Precision (macro): {precision_macro:.4f}")
self.log_message.emit(f"   ├── Recall (macro):    {recall_macro:.4f}")
self.log_message.emit(f"   └── F1-Score (macro):  {f1_macro:.4f}")
```

And include them in the returned `results` dict:
```python
return {
    ...
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    ...
}
```

---

### Task 2: Fix `data_loader.py` — Multi-File Scaler Fitting

**File**: `data_loader.py`

1. **Rename** `fit_scaler_from_first_file` → `fit_scaler_from_samples` (update the call in `trainer.py` too).

2. **Replace the method body** with:

```python
def fit_scaler_from_samples(
    self,
    n_files: int = 8,
    rows_per_file: int = 50000,
    random_state: int = 42,
    log_callback=None
) -> StandardScaler:
    """
    Fit StandardScaler using random samples from multiple files.
    This prevents distribution bias when files are grouped by class.

    Strategy: randomly select min(n_files, total_files) files, read
    up to rows_per_file rows from each, and partial_fit progressively.
    """
    import random

    if not self.file_paths:
        raise DataValidationError("Fayllar yuklanmagan!")

    if log_callback:
        log_callback(f"📏 Scaler ko'p fayldan o'rganilmoqda (distribution bias oldini olish)...")

    self.scaler = StandardScaler()

    # Randomly pick files
    rng = random.Random(random_state)
    n_to_sample = min(n_files, len(self.file_paths))
    sampled_files = rng.sample(self.file_paths, n_to_sample)

    if log_callback:
        log_callback(f"   ├── {n_to_sample} ta fayldan ~{rows_per_file:,} qator namuna olinadi")

    total_samples_used = 0
    for idx, fp in enumerate(sampled_files):
        try:
            # Read first rows_per_file rows only — fast
            df = pd.read_csv(
                fp,
                nrows=rows_per_file,
                low_memory=False
            )
            df.columns = df.columns.str.strip()

            # Check feature columns exist
            missing = [f for f in self.feature_columns if f not in df.columns]
            if missing:
                if log_callback:
                    log_callback(f"   ⚠️ {os.path.basename(fp)} — {len(missing)} feature yo'q, o'tkazildi")
                continue

            feature_data = df[self.feature_columns].copy()
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan).fillna(0)
            feature_data = downcast_dataframe(feature_data)

            self.scaler.partial_fit(feature_data.values)
            total_samples_used += len(feature_data)

            if log_callback:
                log_callback(f"   ├── [{idx+1}/{n_to_sample}] {os.path.basename(fp)} ({len(feature_data):,} qator)")

        except Exception as e:
            if log_callback:
                log_callback(f"   ⚠️ {os.path.basename(fp)}: {str(e)[:100]}")
            continue

    if total_samples_used == 0:
        raise DataValidationError("Scaler uchun yetarli ma'lumot topilmadi!")

    if log_callback:
        log_callback(f"   └── ✅ Scaler tayyor ({total_samples_used:,} namunadan o'rganildi)")

    return self.scaler
```

3. **Keep backward compatibility**: add this alias right after the new method (so existing code doesn't break):

```python
# Backward compatibility alias
def fit_scaler_from_first_file(self, log_callback=None) -> StandardScaler:
    """Deprecated: use fit_scaler_from_samples instead."""
    return self.fit_scaler_from_samples(log_callback=log_callback)
```

4. **Update `trainer.py`** to call the new method:

Find this line in `trainer.py`:
```python
dl.fit_scaler_from_first_file(log_callback=self._emit_log)
```

Replace with:
```python
dl.fit_scaler_from_samples(
    n_files=8,
    rows_per_file=50000,
    log_callback=self._emit_log
)
```

---

### Task 3: Secure Bot Token Management

**File**: `bot_handler.py`

1. **Remove** the hardcoded token (line 41):
```python
# REMOVE THIS LINE:
BOT_TOKEN = "8789775060:AAF8UmdufcsJFCfwJkudz8LV9n5Cz4So_ag"
```

2. **Replace** with environment variable loading. Add at the top of the file (after imports):

```python
# ═══════════════════════════════════════════════════════════
# SECURE TOKEN MANAGEMENT
# ═══════════════════════════════════════════════════════════

# Try loading from .env file first (development convenience)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # python-dotenv optional

BOT_TOKEN = os.environ.get("IOT_SHIELD_BOT_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError(
        "\n"
        "❌ IOT_SHIELD_BOT_TOKEN muhit o'zgaruvchisi sozlanmagan!\n"
        "\n"
        "Yechim:\n"
        "  1. .env fayl yarating va quyidagini qo'shing:\n"
        "     IOT_SHIELD_BOT_TOKEN=your_token_here\n"
        "\n"
        "  2. Yoki terminalda export qiling:\n"
        "     export IOT_SHIELD_BOT_TOKEN='your_token_here'   (Linux/Mac)\n"
        "     set IOT_SHIELD_BOT_TOKEN=your_token_here         (Windows)\n"
        "\n"
        "  3. Tokenni @BotFather dan oling: https://t.me/BotFather\n"
    )
```

3. **Create** `.env.example` file in the project root:

```
# IoT-Shield Configuration
# Copy this file to .env and fill in your values

# Telegram Bot Token (get from @BotFather)
IOT_SHIELD_BOT_TOKEN=your_bot_token_here
```

4. **Create or update** `.gitignore` to ensure `.env` is NEVER committed:

```
# Secrets
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*.pkl
*.pyo
venv/
.venv/

# IDE
.vscode/
.idea/

# Models & logs
models/
logs/
*.log
debug_log.txt

# Session
session.json
```

5. **Update** `requirements.txt` to include `python-dotenv`:

```
python-dotenv>=1.0.0
```

6. **IMPORTANT security note for the user** (add as a comment block at the top of `bot_handler.py`, before any imports):

```python
"""
═══════════════════════════════════════════════════════════════════════
⚠️  XAVFSIZLIK OGOHLANTIRISHI:
═══════════════════════════════════════════════════════════════════════
Agar ilgari BOT_TOKEN kodda ochiq yozilgan bo'lsa va kod git ga
yuborilgan bo'lsa — HOZIROQ eski tokenni @BotFather orqali
bekor qiling va yangisini yarating!

Yangi tokenni yaratish:
  1. Telegram da @BotFather ga yozing
  2. /mybots → botni tanlang → API Token → Revoke current token
  3. Yangi token → .env fayliga qo'ying
═══════════════════════════════════════════════════════════════════════
"""
```

---

## TESTING & VALIDATION

After implementing all changes, please verify:

1. **Syntax check** — run `python -c "import trainer, data_loader, bot_handler"` to ensure no import errors.

2. **Dry-run test** — create a minimal test that:
   - Loads 2-3 small CSV samples (first 10K rows each)
   - Calls `fit_scaler_from_samples(n_files=2, rows_per_file=5000)`
   - Verifies the scaler has non-trivial `mean_` and `scale_` arrays
   - Compares the new scaler stats with the old single-file approach to confirm they differ

3. **Bot token test** — verify that:
   - Running `guard.py` WITHOUT `IOT_SHIELD_BOT_TOKEN` raises the clear error message
   - Running WITH `.env` file works correctly

4. **Produce a summary report** in a new file `docs/PHASE1_CHANGES.md`:
   - What was changed
   - Why it was changed
   - Expected impact on accuracy (3-sentence explanation)
   - How to verify the fix worked

---

## CONSTRAINTS & NON-GOALS

- **DO NOT** change the model algorithm itself (keep `SGDClassifier`). That's Phase 3.
- **DO NOT** add new features yet. That's Phase 2.
- **DO NOT** break backward compatibility with existing `.pkl` models — the inference code in `detector.py` must still work with the old models.
- **DO NOT** touch `ui_design.py` or `detector.py` in this phase (minimal changes only).
- Keep all user-facing logs in Uzbek (to match existing style).
- Preserve the existing code structure, comments, and emoji usage in logs.

## SUCCESS CRITERIA

✅ Training now uses `sample_weight='balanced'` per chunk
✅ Scaler is fitted from ≥5 random files
✅ Macro F1, Precision, Recall are logged and stored in results
✅ Bot token is loaded from environment variable, no hardcoded secrets
✅ `.env.example` and `.gitignore` files exist
✅ Old models still load and run via `detector.py` without errors
✅ `PHASE1_CHANGES.md` summary exists

---

## EXPECTED IMPACT

| Metric | Before | After Phase 1 (expected) |
|--------|--------|--------------------------|
| Accuracy (weighted) | 74.86% | 78-82% |
| F1 (weighted) | 72.40% | 76-80% |
| F1 (macro) | ~40-50% (likely) | 60-70% |
| Security posture | 🔴 Token exposed | 🟢 Secure |

Training time will increase by ~5-10% due to `sample_weight` computation, but this is acceptable.

---

**When done**, reply with:
1. A summary of files changed
2. Key snippets of the critical changes
3. The full content of `PHASE1_CHANGES.md`
4. Confirmation that all 4 success criteria are met
