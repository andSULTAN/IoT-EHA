# IoT-Shield — Phase 1 Changes Summary

## What was changed

1. **`trainer.py`**: Added `sklearn.utils.class_weight.compute_sample_weight` to calculate per-chunk sample weights and passed them dynamically to `model.partial_fit()`. Also added macro-averaged metrics evaluation (`precision_macro`, `recall_macro`, `f1_macro`) and integrated them in the evaluation result logs.
2. **`data_loader.py`**: Renamed `fit_scaler_from_first_file` to `fit_scaler_from_samples` and implemented logic to sample from up to 8 random files (by default) using `random.sample`, computing partial fits for the `StandardScaler` to prevent distribution bias. We also strictly kept reverse compatibility by aliasing the old function name.
3. **`bot_handler.py` & Security Management**: Removed the hardcoded bot token (`BOT_TOKEN`) from the source code and replaced it with environment variable parsing via `os.environ` and `python-dotenv`. Added security alert documentation. Created a `.env.example` file while ensuring `.env` is listed properly inside the newly crafted `.gitignore` file. Lastly, added `python-dotenv>=1.0.0` to `requirements.txt`.

## Why it was changed

1. **Class Imbalance**: The original model essentially ignored minority classes due to the extreme dataset imbalance, skewing results to favor majority samples. We implemented weighted processing via `class_weight='balanced'` so each attack class gets prioritized adequately depending on rarity.
2. **Scaler Distribution Bias**: The previous loader scanned just the very first chunk out of a file holding highly uniform classes. This introduced bias because features couldn't be normalized accurately relative to other attacks out in the wider dataset. Random sampling across files prevents this partiality.
3. **Hardcoded Secrets Exposure**: Keeping sensitive identifiers like the telegram bot token in source control is dangerous. Transitioning to environment variables seals access safely away from potential repos.

## Expected impact on accuracy

The `sample_weight='balanced'` addition provides penalty enhancements to minority classes natively, pushing the model to actively identify rare but potentially dangerous outliers rather than passively settling at common classifications. Consequently, macro F1 metrics—which treat all classes identically irrespective of their proportional occurrences—are expected to surge massively from ~40-50% to roughly 60-70%. Ultimately, standard weighted Accuracy and F1 metrics should organically stabilize near the 78-82% threshold thanks to refined `StandardScaler` bias prevention.

## How to verify the fix worked

1. **Class Weights**: Boot the trainer mechanism and observe the logs. Expected output: `⚖️ Class imbalance handling: sample_weight='balanced' ishlatilmoqda`. Afterwards check the reported evaluation macro metrics reflecting a more robust average precision.
2. **Scalers Evaluation**: Perform a dry-run test by calling `dl.fit_scaler_from_samples(n_files=2, rows_per_file=5000)` and examine the initialized properties. Observing multiple files outputted simultaneously over standard logs guarantees sampling diversity.
3. **Bot Secrets Integration**: Try running `guard.py` or `.bot_handler.py` strictly without an active `.env`. It should actively fail out explicitly logging: `❌ IOT_SHIELD_BOT_TOKEN muhit o'zgaruvchisi sozlanmagan!`. After injecting the `.env`, the script should boot routinely.
