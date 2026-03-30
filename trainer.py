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

# Model saqlash papkasi
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class TrainerThread(QThread):
    """
    SGDClassifier bilan Incremental Learning.
    Har bir faylni chunk-lab o'qib, partial_fit orqali o'qitadi.
    GUI bloklanmasligi uchun alohida threadda ishlaydi.
    """

    # Signallar
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    training_completed = pyqtSignal(dict)
    training_failed = pyqtSignal(str)

    def __init__(self, data_loader=None, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self._is_cancelled = False

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

            dl.scan_all_classes(log_callback=self._emit_log)
            all_classes_encoded = dl.label_encoder.transform(dl.class_names)
            self.progress_updated.emit(5)

            if self._is_cancelled:
                return self._cancel()

            # ═══ 2. SCALER O'RGANISH ═══
            self.log_message.emit("\n" + "─" * 40)
            self.log_message.emit("📏 2-BOSQICH: Scaler o'rganish")
            self.log_message.emit("─" * 40)

            dl.fit_scaler_from_first_file(log_callback=self._emit_log)
            self.progress_updated.emit(8)

            if self._is_cancelled:
                return self._cancel()

            # ═══ 3. MODEL QURILISHI ═══
            self.log_message.emit("\n" + "─" * 40)
            self.log_message.emit("🧠 3-BOSQICH: Model yaratish va o'qitish")
            self.log_message.emit("─" * 40)

            model = SGDClassifier(
                loss='modified_huber',    # Ehtimollik beradi, ko'p klassli
                penalty='l2',
                alpha=1e-4,
                max_iter=1,
                tol=None,
                random_state=42,
                n_jobs=-1,
                warm_start=False,
                class_weight=None,        # partial_fit bilan mos
            )

            self.log_message.emit("   ├── SGDClassifier(loss='modified_huber', class_weight=None)")
            self.log_message.emit("   ├── ℹ️ Har bir chunk shuffle qilinadi (balans uchun)")
            self.log_message.emit(f"   └── Klasslar soni: {len(all_classes_encoded)}")
            self.log_message.emit("")

            # ═══ 4. INCREMENTAL O'QITISH ═══
            total_samples_trained = 0
            per_file_accuracy = []
            test_X = None
            test_y = None
            training_progress_base = 10  # 10% dan boshlaymiz
            training_progress_range = 80  # 80% gacha o'qitish

            for file_idx, file_path in enumerate(file_paths):
                if self._is_cancelled:
                    return self._cancel()

                file_name = os.path.basename(file_path)
                file_start = time.time()

                self.log_message.emit(f"📂 [{file_idx+1}/{total_files}] {file_name}")

                file_samples = 0
                chunk_count = 0
                is_last_file = (file_idx == total_files - 1)

                for X_chunk, y_chunk in dl.stream_file_chunks(file_path, chunksize=100000):
                    chunk_count += 1

                    try:
                        # Oxirgi faylning oxirgi chunkidan test set ajratish
                        if is_last_file and test_X is None and chunk_count == 1:
                            split_idx = int(len(X_chunk) * 0.8)
                            if split_idx > 100:
                                test_X = X_chunk[split_idx:]
                                test_y = y_chunk[split_idx:]
                                X_chunk = X_chunk[:split_idx]
                                y_chunk = y_chunk[:split_idx]

                        # Chunk ichidagi ma'lumotlarni shuffle qilish (balans uchun)
                        if len(X_chunk) > 0:
                            X_chunk, y_chunk = sklearn_shuffle(
                                X_chunk, y_chunk, random_state=42 + chunk_count
                            )

                        # Partial fit!
                        if len(X_chunk) > 0:
                            model.partial_fit(X_chunk, y_chunk, classes=all_classes_encoded)
                            file_samples += len(X_chunk)

                    except Exception as chunk_err:
                        self.log_message.emit(
                            f"   ⚠️ Chunk #{chunk_count} xato: {str(chunk_err)[:120]}"
                        )
                        continue

                total_samples_trained += file_samples
                file_time = time.time() - file_start

                # Har bir fayldan keyin mini-accuracy hisoblash (agar test mavjud)
                file_acc_str = ""
                if test_X is not None and len(test_X) > 0:
                    y_pred_test = model.predict(test_X)
                    current_acc = accuracy_score(test_y, y_pred_test)
                    per_file_accuracy.append(current_acc)
                    file_acc_str = f" | acc: {current_acc:.4f}"

                self.log_message.emit(
                    f"   ✅ {file_samples:,} qator o'qitildi ({file_time:.1f}s){file_acc_str}"
                )
                self.log_message.emit(
                    f"   📊 Umumiy progress: {file_idx+1}/{total_files}"
                )

                # Progress bar yangilash
                progress = training_progress_base + int(
                    (file_idx + 1) / total_files * training_progress_range
                )
                self.progress_updated.emit(min(progress, 90))

                # Memory tozalash
                gc.collect()

            self.log_message.emit(f"\n✅ Jami {total_samples_trained:,} ta qator o'qitildi")

            if self._is_cancelled:
                return self._cancel()

            # ═══ 5. BAHOLASH ═══
            self.log_message.emit("\n" + "─" * 40)
            self.log_message.emit("📊 4-BOSQICH: Model baholash")
            self.log_message.emit("─" * 40)
            self.progress_updated.emit(92)

            results = self._evaluate_model(
                model, test_X, test_y, dl.class_names, dl.label_encoder
            )
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

        self.log_message.emit(f"   ├── Precision: {precision:.4f}")
        self.log_message.emit(f"   ├── Recall:    {recall:.4f}")
        self.log_message.emit(f"   └── F1-Score:  {f1:.4f}")

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
        pkl_path = os.path.join(MODELS_DIR, f"iot_shield_sgd_{timestamp}.pkl")
        joblib.dump(model, pkl_path, compress=3)
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
            "model_type": "SGDClassifier (Incremental Learning)",
            "created_at": datetime.now().isoformat(),
            "feature_names": dl.feature_columns,
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
