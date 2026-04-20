"""
IoT-Shield AI Trainer - Data Loader Module
CICIOT2023 dataset uchun:
- Fayl yuklash va ko'rish (GUI stat kartochkalar uchun)
- Incremental Learning uchun streaming data pipeline
- Memory-optimized chunk-based reading (float64 → float32)
"""

import os
import gc
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Optional, Dict, Generator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from feature_engineering import add_derived_features_df, DERIVED_FEATURES, FEATURE_ENG_VERSION

# ═══ CICIOT2023 datasetidagi 39 ta parametr ═══
CICIOT2023_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration",
    "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]

KNOWN_LABEL_COLUMNS = ["label", "Label", "LABEL", "class", "Class", "attack", "Attack", "category"]


class DataValidationError(Exception):
    pass


class InsufficientDiversityError(Exception):
    pass


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Memory optimization: float64 → float32, int64 → int32."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int32)
    return df


class DataLoader:
    """
    CICIOT2023 dataset uchun Data Loader.
    - GUI uchun: fayl yuklash, statistika
    - Training uchun: streaming chunks, incremental data pipeline
    """

    def __init__(self):
        self.dataframe: Optional[pd.DataFrame] = None
        self.loaded_files: List[str] = []       # Fayl nomlari
        self.file_paths: List[str] = []         # To'liq fayl yo'llari
        self.feature_columns: List[str] = []
        self.label_column: Optional[str] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.class_names: List[str] = []
        self.total_rows: int = 0

    def clear_all(self):
        """Barcha yuklangan ma'lumotlarni tozalash."""
        self.dataframe = None
        self.loaded_files = []
        self.file_paths = []
        self.feature_columns = []
        self.label_column = None
        self.label_encoder = None
        self.scaler = None
        self.class_names = []
        self.total_rows = 0
        gc.collect()

    # ═══════════════════════════════════════════════════════
    # GUI uchun: Fayllarni yuklash va preview
    # ═══════════════════════════════════════════════════════

    def load_csv_files(self, file_paths: List[str], log_callback=None) -> pd.DataFrame:
        """
        CSV fayllarni yuklash (GUI statistika ko'rsatish uchun).
        Faqat birinchi 5000 qatorni o'qiydi (preview uchun).
        To'liq ma'lumot training vaqtida chunk-lab o'qiladi.
        """
        new_files = []

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            abs_path = os.path.abspath(file_path)

            if file_name in self.loaded_files:
                if log_callback:
                    log_callback(f"   ⏭️ '{file_name}' allaqachon yuklangan")
                continue

            if log_callback:
                log_callback(f"📂 Yuklanmoqda: {file_name}...")

            _, ext = os.path.splitext(file_name)
            if ext.lower() not in ['.csv', '.txt', '.tsv', '.data']:
                raise DataValidationError(
                    f"Xato: '{file_name}' — CSV formatida emas!\n"
                    f"Qo'llab-quvvatlanadigan: .csv, .txt, .tsv, .data"
                )

            try:
                # Faqat qator sonini hisoblash (tez)
                row_count = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore')) - 1

                # Preview uchun birinchi qatorlarni o'qish
                preview = pd.read_csv(file_path, nrows=5000, low_memory=False)
                preview.columns = preview.columns.str.strip()

            except Exception as e:
                if log_callback:
                    log_callback(f"❌ Xato: {file_name} - {str(e)}")
                raise DataValidationError(f"'{file_name}' o'qib bo'lmadi: {str(e)}")

            if len(preview.columns) == 0 or len(preview) == 0:
                raise DataValidationError(f"'{file_name}' bo'sh yoki noto'g'ri formatda")

            if log_callback:
                log_callback(f"   ├── Ustunlar: {len(preview.columns)}")
                log_callback(f"   ├── Qatorlar: {row_count:,}")
                log_callback(f"   └── ✅ {file_name} tayyor!")

            self.loaded_files.append(file_name)
            self.file_paths.append(abs_path)
            self.total_rows += row_count
            new_files.append(file_name)

            # Preview dataframeni saqlash (faqat birinchi fayldan)
            if self.dataframe is None:
                self.dataframe = downcast_dataframe(preview)
            else:
                # Faqat row count ni yangilash, butun faylni yuklash shart emas
                pass

        if not new_files:
            if log_callback:
                log_callback("ℹ️ Yangi fayllar topilmadi")
            return self.dataframe

        # Feature va label ustunlarini aniqlash (preview dan)
        self._auto_detect_columns(log_callback)

        if log_callback:
            log_callback(f"\n📊 Jami: {self.total_rows:,} qator ({len(self.loaded_files)} fayl)")

        return self.dataframe

    def load_from_directory(self, directory_path: str, log_callback=None) -> pd.DataFrame:
        """Papkadagi barcha CSV fayllarni yuklash."""
        if not os.path.isdir(directory_path):
            raise DataValidationError(f"'{directory_path}' papkasi topilmadi!")

        csv_files = [
            os.path.join(directory_path, f)
            for f in sorted(os.listdir(directory_path))
            if f.lower().endswith('.csv')
        ]

        if not csv_files:
            raise DataValidationError(f"Papkada CSV fayllar topilmadi!")

        if log_callback:
            log_callback(f"📁 '{os.path.basename(directory_path)}' — {len(csv_files)} ta CSV fayl\n")

        return self.load_csv_files(csv_files, log_callback)

    # ═══════════════════════════════════════════════════════
    # INCREMENTAL LEARNING uchun: Streaming pipeline
    # ═══════════════════════════════════════════════════════

    def scan_all_classes(self, log_callback=None) -> List[str]:
        """
        Barcha fayllarni skanlab, noyob klasslarni aniqlash.
        partial_fit uchun classes parametri kerak.
        Faqat label ustunini o'qiydi — tez ishlaydi.
        """
        if not self.file_paths:
            raise DataValidationError("Fayllar yuklanmagan!")

        if self.label_column is None:
            raise DataValidationError("Label ustuni topilmadi!")

        if log_callback:
            log_callback("🔍 Barcha klasslar skanlanmoqda...")

        all_classes = set()
        for i, fp in enumerate(self.file_paths):
            try:
                # Faqat label ustunini o'qish (memory efficient)
                label_col = pd.read_csv(
                    fp, usecols=[self.label_column],
                    low_memory=False, dtype=str
                )
                unique = set(label_col[self.label_column].dropna().unique())
                all_classes.update(unique)

                if log_callback and (i + 1) % 10 == 0:
                    log_callback(f"   ├── {i+1}/{len(self.file_paths)} fayl skanlandi...")

            except Exception as e:
                if log_callback:
                    log_callback(f"   ⚠️ {os.path.basename(fp)}: {e}")

        self.class_names = sorted(list(all_classes))

        # LabelEncoder yaratish
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)

        if log_callback:
            log_callback(f"   └── ✅ {len(self.class_names)} ta klass topildi")
            for i, name in enumerate(self.class_names):
                log_callback(f"       {i}: '{name}'")

        return self.class_names

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
                
                # Apply feature engineering BEFORE scaler fitting
                feature_data = add_derived_features_df(feature_data)
                
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

    # Backward compatibility alias
    def fit_scaler_from_first_file(self, log_callback=None) -> StandardScaler:
        """Deprecated: use fit_scaler_from_samples instead."""
        return self.fit_scaler_from_samples(log_callback=log_callback)

    def stream_file_chunks(
        self, file_path: str, chunksize: int = 100000
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Bitta faylni chunk-lab o'qish generatori.
        Har bir chunk uchun (X_scaled, y_encoded) qaytaradi.

        - float64 → float32 downcast
        - NaN/Inf → 0 ga almashtirish
        - LabelEncoder bilan kodlash
        - StandardScaler bilan normallash
        """
        if self.label_encoder is None or self.scaler is None:
            raise DataValidationError("Avval scan_all_classes va fit_scaler ni chaqiring!")

        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            chunk.columns = chunk.columns.str.strip()

            # Label ustuni bormi tekshirish
            if self.label_column not in chunk.columns:
                continue

            # Feature data
            X = chunk[self.feature_columns].copy()
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # === FEATURE ENGINEERING ===
            X = add_derived_features_df(X)
            
            X = downcast_dataframe(X)

            # Label data — noma'lum labellarni filtrlash
            raw_labels = chunk[self.label_column].astype(str)
            known_mask = raw_labels.isin(self.class_names)

            if known_mask.sum() == 0:
                continue

            X = X[known_mask]
            y = self.label_encoder.transform(raw_labels[known_mask])

            # StandardScaler transform
            X_scaled = self.scaler.transform(X.values).astype(np.float32)

            yield X_scaled, y

    # ═══════════════════════════════════════════════════════
    # Column detection
    # ═══════════════════════════════════════════════════════

    def _auto_detect_columns(self, log_callback=None):
        """CICIOT2023 39 ta parametrni va Label ustunini aniqlash."""
        df = self.dataframe

        # 1. Label ustunini aniqlash
        self.label_column = None
        for col_name in KNOWN_LABEL_COLUMNS:
            if col_name in df.columns:
                self.label_column = col_name
                break

        if self.label_column is None:
            object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if object_cols:
                self.label_column = object_cols[-1]

        if log_callback:
            if self.label_column:
                unique_count = df[self.label_column].nunique()
                log_callback(f"\n🏷️  Label ustuni: '{self.label_column}' ({unique_count} ta klass)")
            else:
                log_callback(f"\n❌ Label ustuni topilmadi!")

        # 2. CICIOT2023 39 ta parametrdan mavjudlarini aniqlash
        existing_columns = set(df.columns)
        ciciot_found = [f for f in CICIOT2023_FEATURES if f in existing_columns]

        if len(ciciot_found) >= 20:
            self.feature_columns = sorted(ciciot_found)
            if log_callback:
                log_callback(f"📐 CICIOT2023 parametrlari: {len(ciciot_found)}/{len(CICIOT2023_FEATURES)} topildi ✅")
                missing = set(CICIOT2023_FEATURES) - set(ciciot_found)
                if missing:
                    log_callback(f"   ├── ⚠️ Topilmagan: {', '.join(sorted(missing))}")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.label_column and self.label_column in numeric_cols:
                numeric_cols.remove(self.label_column)
            self.feature_columns = sorted(numeric_cols)
            if log_callback:
                log_callback(f"📐 Feature ustunlari: {len(self.feature_columns)} ta raqamli ustun")

        if len(self.feature_columns) == 0:
            raise DataValidationError(
                "Faylda raqamli ustunlar topilmadi!\n"
                "Model o'qitish uchun kamida bir nechta raqamli ustun kerak."
            )

    # ═══════════════════════════════════════════════════════
    # Balance check & summary
    # ═══════════════════════════════════════════════════════

    def check_data_balance(self, log_callback=None) -> Tuple[bool, dict]:
        """Ma'lumotlar balansini tekshirish (preview asosida)."""
        if self.dataframe is None:
            raise DataValidationError("Ma'lumotlar yuklanmagan!")

        if self.label_column is None or self.label_column not in self.dataframe.columns:
            if log_callback:
                log_callback(f"\nℹ️ Label ustuni topilmadi — balans tekshiruvi o'tkazib yuborildi")
            return True, {}

        class_distribution = self.dataframe[self.label_column].value_counts().to_dict()
        unique_classes = len(class_distribution)

        if log_callback:
            log_callback(f"\n📈 Ma'lumotlar balansi (preview):")
            log_callback(f"   ├── Klasslar soni: {unique_classes}")
            for cls, count in sorted(class_distribution.items(), key=lambda x: -x[1])[:15]:
                percentage = (count / len(self.dataframe)) * 100
                log_callback(f"   ├── {cls}: {count:,} ({percentage:.1f}%)")
            if unique_classes > 15:
                log_callback(f"   ├── ... va yana {unique_classes - 15} ta klass")

        if unique_classes <= 1:
            raise InsufficientDiversityError(
                "Datasetda faqat 1 ta klass.\nKamida 2 xil klass kerak."
            )

        if log_callback:
            log_callback(f"   └── ✅ Ma'lumotlar balansi qoniqarli")

        return True, class_distribution

    def get_features_and_labels(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Preview uchun features va labels."""
        if self.dataframe is None:
            raise DataValidationError("Ma'lumotlar yuklanmagan!")
        X = self.dataframe[self.feature_columns].copy()
        y = None
        if self.label_column and self.label_column in self.dataframe.columns:
            y = self.dataframe[self.label_column].copy()
        return X, y

    def get_all_feature_names(self) -> List[str]:
        """Return raw + derived feature names in the exact training order."""
        return list(self.feature_columns) + list(DERIVED_FEATURES)

    def get_summary(self) -> dict:
        """Yuklangan ma'lumotlar haqida qisqacha ma'lumot."""
        total_size_mb = 0
        for fp in self.file_paths:
            try:
                total_size_mb += os.path.getsize(fp) / (1024 * 1024)
            except OSError:
                pass

        return {
            "loaded_files": self.loaded_files,
            "total_rows": self.total_rows,
            "total_columns": len(self.feature_columns) + (1 if self.label_column else 0),
            "feature_count": len(self.feature_columns),
            "derived_feature_count": len(DERIVED_FEATURES),
            "total_feature_count": len(self.feature_columns) + len(DERIVED_FEATURES),
            "feature_eng_version": FEATURE_ENG_VERSION,
            "label_column": self.label_column or "—",
            "unique_classes": len(self.class_names) if self.class_names else 0,
            "memory_usage_mb": round(total_size_mb, 2),
        }

    def stream_all_files_round_robin(
        self,
        chunksize: int = 100000,
        shuffle_chunks: bool = True,
        random_state: int = 42,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Interleave chunks from ALL files — prevents catastrophic forgetting.
        """
        from sklearn.utils import shuffle as sklearn_shuffle

        file_gens = []
        for fp in self.file_paths:
            file_gens.append(self.stream_file_chunks(fp, chunksize=chunksize))

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

    def build_stratified_test_set(
        self,
        rows_per_file: int = 2000,
        max_rows_per_class: int = 5000,
        random_state: int = 42,
        log_callback=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a test set that samples across ALL files AND balances classes.
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

                raw_labels = df[self.label_column].astype(str)
                known_mask = raw_labels.isin(self.class_names)
                df = df[known_mask]

                if len(df) == 0:
                    continue

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
