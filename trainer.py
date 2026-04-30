"""
IoT-Shield AI Trainer - Incremental Training Engine
SGDClassifier + partial_fit orqali katta datasetlarni
bo'laklab o'qitish (Incremental / Online Learning).
24 GB RAM uchun optimallashtirilgan.
"""

import os
import time
import gc
import numpy as np
from sklearn.utils import shuffle as sklearn_shuffle
import joblib
from PyQt6.QtCore import QThread, pyqtSignal

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight
from ensemble import IncrementalEnsemble
from hyperparameter_tuner import tune_sgd_hyperparameters
from benchmark import evaluate_model_full, compare_models
from feature_engineering import DERIVED_FEATURES, FEATURE_ENG_VERSION

# Model saqlash papkasi
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class TrainerThread(QThread):
    """
    SGDClassifier bilan Incremental Learning.
    Har bir faylni chunk-lab o'qib, partial_fit orqali o'qitadi.
    GUI bloklanmasligi uchun alohida threadda ishlaydi.
    """
    EPOCHS_PER_CHUNK = 3
    USE_ENSEMBLE = True
    USE_HYPERPARAMETER_TUNING = True
    HP_TUNING_TIME_BUDGET = 300.0  # 5 minutes
    USE_CALIBRATION = False

    # Signallar
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    training_completed = pyqtSignal(dict)
    training_failed = pyqtSignal(str)

    def __init__(self, data_loader=None, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self._is_cancelled = False

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

    def run(self):
        """Incremental o'qitish jarayoni."""
        try:
            start_time = time.time()
            dl = self.data_loader
            file_paths = dl.file_paths
            total_files = len(file_paths)

            self.log_message.emit("🚀 Incremental Learning boshlandi")
            self.log_message.emit(f"   ├── Fayllar: {total_files} ta")
            self.log_message.emit(f"   ├── Features: {len(dl.feature_columns)} ta")
            self.log_message.emit(f"   ├── Model: SGDClassifier (partial_fit)")
            self.log_message.emit(f"   └── Chunk size: 100,000 qator")
            self.progress_updated.emit(2)

            if self._is_cancelled:
                return self._cancel()

            # ═══ 1. BARCHA KLASSLARNI SKANLASH ═══
            self.log_message.emit("\n" + "─" * 40)
            self.log_message.emit("📋 1-BOSQICH: Klasslarni aniqlash")
            self.log_message.emit("─" * 40)

            dl.use_grouping = True
            dl.scan_all_classes(log_callback=self._emit_log)
            all_classes_encoded = dl.label_encoder.transform(dl.class_names)
            self.log_message.emit(f"   ├── 🗂️  Class grouping: 34 → {len(dl.class_names)} sinf")
            self.progress_updated.emit(5)

            if self._is_cancelled:
                return self._cancel()

            # ═══ 2. SCALER O'RGANISH ═══
            self.log_message.emit("\n" + "─" * 40)
            self.log_message.emit("📏 2-BOSQICH: Scaler o'rganish")
            self.log_message.emit("─" * 40)

            dl.fit_scaler_from_samples(
                n_files=8,
                rows_per_file=50000,
                log_callback=self._emit_log
            )
            self.progress_updated.emit(8)

            if self._is_cancelled:
                return self._cancel()

            # ═══ 2.75. STRATIFIED TEST SET (o'qitishdan OLDIN — data leakage oldini olish) ═══
            self.log_message.emit("\n" + "─" * 40)
            self.log_message.emit("🎯 3-BOSQICH: Stratified test set yaratilmoqda (o'qitishdan oldin)")
            self.log_message.emit("─" * 40)

            test_X, test_y = dl.build_stratified_test_set(
                rows_per_file=2000,
                max_rows_per_class=3000,
                log_callback=self._emit_log,
            )
            self.progress_updated.emit(10)

            if self._is_cancelled:
                return self._cancel()

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

            # ═══ 3. MODEL QURILISHI ═══
            self.log_message.emit("\n" + "─" * 40)
            self.log_message.emit("🧠 3-BOSQICH: Model yaratish va o'qitish")
            self.log_message.emit("─" * 40)

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
                    warm_start=False,
                    class_weight=None,
                )

            self.log_message.emit(f"   ├── Model parameters: {best_sgd_params}")
            self.log_message.emit("   ├── ℹ️ Har bir chunk shuffle qilinadi (balans uchun)")
            self.log_message.emit("   ⚖️ Class imbalance handling: sample_weight='balanced' ishlatilmoqda")
            self.log_message.emit(f"   └── Klasslar soni: {len(all_classes_encoded)}")
            self.log_message.emit("")

            # ═══ 4. INCREMENTAL O'QITISH ═══
            total_samples_trained = 0
            per_file_accuracy = []
            training_progress_base = 10  # 10% dan boshlaymiz
            training_progress_range = 80  # 80% gacha o'qitish

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

            if self._is_cancelled:
                return self._cancel()

            # ═══ 5. BAHOLASH (RIGOROUS) ═══
            self.log_message.emit("\n" + "─" * 40)
            self.log_message.emit("📊 4-BOSQICH: Rigorous Benchmarking")
            self.log_message.emit("─" * 40)
            self.progress_updated.emit(92)

            if self.USE_CALIBRATION and test_X is not None and len(test_X) > 0:
                from sklearn.calibration import CalibratedClassifierCV
                self.log_message.emit("\n📐 Model calibration (isotonic)...")
                calib_split = int(len(test_X) * 0.9)
                cal_X, cal_y = test_X[calib_split:], test_y[calib_split:]
                test_X, test_y = test_X[:calib_split], test_y[:calib_split]
                try:
                    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
                    calibrated.fit(cal_X, cal_y)
                    model = calibrated
                    self.log_message.emit("   └── ✅ Calibration yakunlandi")
                except Exception as e:
                    self.log_message.emit(f"   ⚠️ Calibration xato: {e}")

            if test_X is not None and len(test_X) > 0:
                ensemble_eval = evaluate_model_full(
                    model, test_X, test_y, dl.class_names,
                    model_name="Ensemble" if self.USE_ENSEMBLE else "SGD"
                )
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
                bench_dir = os.path.join(os.path.dirname(MODELS_DIR), "benchmarks")
                bench_md = compare_models(all_benchmarks, bench_dir)
                self.log_message.emit(f"\n📄 Benchmark report: {bench_md}")
                self.log_message.emit(f"\n🏆 FINAL METRICS:")
                self.log_message.emit(f"   ├── Accuracy:        {ensemble_eval['accuracy']:.4f}")
                self.log_message.emit(f"   ├── F1 (weighted):   {ensemble_eval['f1_weighted']:.4f}")
                self.log_message.emit(f"   ├── F1 (macro):      {ensemble_eval['f1_macro']:.4f}")
                self.log_message.emit(f"   ├── Latency:         {ensemble_eval['single_predict_latency_ms']:.3f} ms/sample")
                self.log_message.emit(f"   └── Throughput:      {ensemble_eval['samples_per_second']:,} samples/sec")
                results = ensemble_eval
            else:
                results = {"status": "no_test_data"}

            results['per_file_accuracy'] = per_file_accuracy
            results['total_files'] = total_files
            results['total_samples'] = total_samples_trained
            self.progress_updated.emit(95)

            # ═══ 6. SAQLASH ═══
            self.log_message.emit("\n" + "─" * 40)
            self.log_message.emit("💾 5-BOSQICH: Model saqlash")
            self.log_message.emit("─" * 40)

            model_paths = self._save_model(model, dl)
            results['model_paths'] = model_paths
            self.progress_updated.emit(98)

            # ═══ 7. YAKUNLASH ═══
            total_time = time.time() - start_time
            results['total_time'] = total_time
            self.log_message.emit(f"\n⏱️ Jami vaqt: {total_time:.1f} soniya")
            self.progress_updated.emit(100)

            self.training_completed.emit(results)

        except Exception as e:
            import traceback
            self.log_message.emit(f"\n❌ Xato tafsiloti:\n{traceback.format_exc()}")
            self.training_failed.emit(f"O'qitishda xato: {str(e)}")

    def _evaluate_model(self, model, test_X, test_y, class_names, label_encoder) -> dict:
        """Model natijalarini baholash."""
        if test_X is None or len(test_X) == 0:
            self.log_message.emit("⚠️ Test ma'lumotlari topilmadi — baholash o'tkazib yuborildi")
            return {
                "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0,
                "precision_macro": 0, "recall_macro": 0, "f1_macro": 0,
                "confusion_matrix": np.array([]), "class_names": class_names,
                "status": "no_test_data"
            }

        self.log_message.emit(f"   ├── Test namunalar: {len(test_X):,}")

        y_pred = model.predict(test_X)

        # Accuracy
        accuracy = accuracy_score(test_y, y_pred)
        self.log_message.emit(f"   ├── Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Precision, Recall, F1
        precision = precision_score(test_y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(test_y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(test_y, y_pred, average='weighted', zero_division=0)

        # Macro-averaged (treats all classes equally — important for imbalanced data)
        precision_macro = precision_score(test_y, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(test_y, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(test_y, y_pred, average='macro', zero_division=0)

        self.log_message.emit(f"   ├── Precision: {precision:.4f}")
        self.log_message.emit(f"   ├── Recall:    {recall:.4f}")
        self.log_message.emit(f"   ├── F1-Score:  {f1:.4f}")
        self.log_message.emit(f"   ├── Precision (macro): {precision_macro:.4f}")
        self.log_message.emit(f"   ├── Recall (macro):    {recall_macro:.4f}")
        self.log_message.emit(f"   └── F1-Score (macro):  {f1_macro:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(test_y, y_pred)

        # Classification Report
        self.log_message.emit("\n📋 Classification Report:")

        # Test ma'lumotlaridagi haqiqiy klasslar
        unique_test_labels = sorted(set(test_y) | set(y_pred))
        test_class_names = []
        for lbl in unique_test_labels:
            if lbl < len(class_names):
                test_class_names.append(class_names[lbl])
            else:
                test_class_names.append(str(lbl))

        report = classification_report(
            test_y, y_pred,
            labels=unique_test_labels,
            target_names=test_class_names,
            zero_division=0
        )
        for line in report.split('\n'):
            if line.strip():
                self.log_message.emit(f"   {line}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "confusion_matrix": cm,
            "class_names": class_names,
            "y_test": test_y,
            "y_pred": y_pred,
            "model": model,
            "status": "completed",
        }

    def _save_model(self, model, dl) -> dict:
        """O'qitilgan modelni saqlash (.pkl + metadata)."""
        import json
        from datetime import datetime

        os.makedirs(MODELS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}

        # 1. Model (.pkl)
        if hasattr(model, "save") and hasattr(model, "models"):
            pkl_path = os.path.join(MODELS_DIR, f"iot_shield_ensemble_{timestamp}.pkl")
            model.save(pkl_path)
            metadata_type = "IncrementalEnsemble (3-model soft voting)"
            ensemble_config = model.summary()
        else:
            pkl_path = os.path.join(MODELS_DIR, f"iot_shield_sgd_{timestamp}.pkl")
            joblib.dump(model, pkl_path, compress=3)
            metadata_type = "SGDClassifier (Incremental Learning)"
            ensemble_config = None

        pkl_size = os.path.getsize(pkl_path) / (1024 * 1024)
        self.log_message.emit(f"   ├── 📦 Model: {os.path.basename(pkl_path)} ({pkl_size:.2f} MB)")
        paths["pkl"] = pkl_path

        # 2. Scaler
        if dl.scaler is not None:
            scaler_path = os.path.join(MODELS_DIR, f"scaler_{timestamp}.pkl")
            joblib.dump(dl.scaler, scaler_path)
            paths["scaler"] = scaler_path
            self.log_message.emit(f"   ├── 📏 Scaler: {os.path.basename(scaler_path)}")

        # 3. LabelEncoder
        if dl.label_encoder is not None:
            encoder_path = os.path.join(MODELS_DIR, f"label_encoder_{timestamp}.pkl")
            joblib.dump(dl.label_encoder, encoder_path)
            paths["label_encoder"] = encoder_path
            self.log_message.emit(f"   ├── 🏷️ Encoder: {os.path.basename(encoder_path)}")

        # 4. Metadata
        metadata = {
            "model_type": metadata_type,
            "ensemble_config": ensemble_config,
            "created_at": datetime.now().isoformat(),
            "feature_names": dl.feature_columns,
            "feature_eng_version": FEATURE_ENG_VERSION,
            "raw_feature_names": dl.feature_columns,
            "derived_feature_names": list(DERIVED_FEATURES),
            "all_feature_names": dl.get_all_feature_names(),
            "total_features": len(dl.feature_columns) + len(DERIVED_FEATURES),
            "epochs_per_chunk": self.EPOCHS_PER_CHUNK,
            "class_names": dl.class_names,
            "n_features": len(dl.feature_columns),
            "n_classes": len(dl.class_names),
            "total_files": len(dl.file_paths),
            "total_rows": dl.total_rows,
            "files": {k: os.path.basename(v) for k, v in paths.items()},
        }

        meta_path = os.path.join(MODELS_DIR, f"metadata_{timestamp}.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        paths["metadata"] = meta_path

        self.log_message.emit(f"   └── ✅ Saqlandi: '{MODELS_DIR}'")
        self.log_message.emit("")
        self.log_message.emit("   ℹ️ Raspberry Pi uchun: .pkl fayllarini Python+Joblib")
        self.log_message.emit("      orqali yuklash va predict qilish mumkin.")

        return paths

    def _emit_log(self, msg):
        """Thread-safe log yuborish."""
        self.log_message.emit(msg)

    def _cancel(self):
        self.log_message.emit("⛔ O'qitish bekor qilindi.")

    def cancel(self):
        self._is_cancelled = True
        self.log_message.emit("⏳ O'qitish bekor qilinmoqda...")
