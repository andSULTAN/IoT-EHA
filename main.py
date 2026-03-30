"""
IoT-Shield AI Trainer - Main Module
Barcha modullarni birlashtiruvchi asosiy fayl.
"""

import sys
import os
import json
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui_design import MainWindow, COLORS
from data_loader import DataLoader, DataValidationError, InsufficientDiversityError
from trainer import TrainerThread

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Session fayli — yuklangan fayl yo'llarini saqlash
SESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session.json")


def format_data_size(size_bytes: float) -> str:
    """Ma'lumot hajmini MB yoki GB da formatlash."""
    if size_bytes < 1:
        return "0 MB"
    mb = size_bytes
    if mb < 1024:
        return f"{mb:.1f} MB"
    else:
        gb = mb / 1024
        return f"{gb:.2f} GB"


class IoTShieldApp:
    """IoT-Shield AI Trainer ilovasi kontrolleri."""

    def __init__(self):
        self.app = QApplication(sys.argv)

        # App-wide font
        font = QFont("Segoe UI", 10)
        self.app.setFont(font)

        # Dark mode palette
        self.app.setStyle("Fusion")

        # Oynani yaratish
        self.window = MainWindow()

        # Data loader
        self.data_loader = DataLoader()

        # Trainer thread
        self.trainer_thread = None

        # Signallarni ulash
        self._connect_signals()

        self.window.append_log("╔══════════════════════════════════════════╗")
        self.window.append_log("║  🛡️  IoT-Shield AI Trainer v1.0.0        ║")
        self.window.append_log("║  CICIOT2023 — NIDS Dashboard             ║")
        self.window.append_log("╚══════════════════════════════════════════╝")
        self.window.append_log("")

        # Avvalgi sessiyani tiklash
        self._restore_session()

    def _connect_signals(self):
        """Tugmalar va signallarni ulash."""
        # Tugmalar
        self.window.btn_load_csv.clicked.connect(self._on_load_csv)
        self.window.btn_load_dir.clicked.connect(self._on_load_directory)
        self.window.btn_clear.clicked.connect(self._on_clear_data)
        self.window.btn_train.clicked.connect(self._on_train_model)

        # Menu actions
        self.window.action_open_csv.triggered.connect(self._on_load_csv)
        self.window.action_open_dir.triggered.connect(self._on_load_directory)
        self.window.action_exit.triggered.connect(self.window.close)
        self.window.action_about.triggered.connect(self._show_about)

    def _on_load_csv(self):
        """CSV fayllarni tanlash va yuklash."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.window,
            "CICIOT2023 CSV Fayllarni Tanlang",
            "",
            "CSV Fayllar (*.csv);;Barcha fayllar (*.*)"
        )

        if not file_paths:
            return

        self._load_data(file_paths)

    def _on_load_directory(self):
        """MERGED_CSV papkasini tanlash va yuklash."""
        directory = QFileDialog.getExistingDirectory(
            self.window,
            "MERGED_CSV Papkasini Tanlang",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if not directory:
            return

        self._load_data_from_directory(directory)

    def _load_data(self, file_paths: list):
        """Ma'lumotlarni yuklash va tekshirish (davomli qo'shish)."""
        self.window.set_progress(0)
        self.window.append_log("\n" + "─" * 40)
        self.window.append_log(f"🔄 {len(file_paths)} ta fayl yuklanmoqda...\n")

        try:
            # CSV fayllarni yuklash
            self.data_loader.load_csv_files(
                file_paths,
                log_callback=self.window.append_log
            )

            # Ma'lumotlar balansini tekshirish
            is_balanced, class_dist = self.data_loader.check_data_balance(
                log_callback=self.window.append_log
            )

            self._update_stat_cards()

            # Train va Clear tugmalarini aktivlashtirish
            self.window.btn_train.setEnabled(True)
            self.window.btn_clear.setEnabled(True)

            # Sessiyani saqlash
            self._save_session(file_paths)

            # Status bar
            summary = self.data_loader.get_summary()
            self.window.statusBar().showMessage(
                f"✅ {summary['total_rows']:,} qator  |  "
                f"{format_data_size(summary['memory_usage_mb'])}  |  "
                f"{summary['total_columns']} ustun  |  "
                f"{len(summary['loaded_files'])} fayl"
            )

            self.window.append_log("\n✅ Ma'lumotlar muvaffaqiyatli yuklandi!")
            self.window.append_log("   ➜ Yana CSV qo'shishingiz yoki 'Modelni O'qitish' ni bosishingiz mumkin.")

        except DataValidationError as e:
            self._show_error("Ma'lumot Validatsiyasi Xatosi", str(e))
            self.window.append_log(f"\n❌ XATO: {str(e)}")
            self.window.btn_train.setEnabled(False)

        except InsufficientDiversityError as e:
            self._show_warning("Ma'lumotlar Balansi Ogohlantirishi", str(e))
            self.window.append_log(f"\n⚠️ OGOHLANTIRISH: {str(e)}")
            self.window.btn_train.setEnabled(False)

        except Exception as e:
            self._show_error("Kutilmagan Xato", f"Kutilmagan xato yuz berdi:\n{str(e)}")
            self.window.append_log(f"\n❌ Kutilmagan xato: {str(e)}")
            self.window.btn_train.setEnabled(False)

    def _load_data_from_directory(self, directory: str):
        """Papkadan ma'lumotlarni yuklash (davomli qo'shish)."""
        self.window.set_progress(0)
        self.window.append_log("\n" + "─" * 40)
        self.window.append_log(f"📁 Papka yuklanmoqda: {os.path.basename(directory)}\n")

        try:
            self.data_loader.load_from_directory(
                directory,
                log_callback=self.window.append_log
            )

            is_balanced, class_dist = self.data_loader.check_data_balance(
                log_callback=self.window.append_log
            )

            self._update_stat_cards()
            self.window.btn_train.setEnabled(True)
            self.window.btn_clear.setEnabled(True)

            # Sessiyani saqlash (papkadagi fayllar)
            csv_paths = [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith('.csv')
            ]
            self._save_session(csv_paths)

            summary = self.data_loader.get_summary()
            self.window.statusBar().showMessage(
                f"✅ {summary['total_rows']:,} qator  |  "
                f"{format_data_size(summary['memory_usage_mb'])}  |  "
                f"{summary['total_columns']} ustun  |  "
                f"{len(summary['loaded_files'])} fayl"
            )

            self.window.append_log("\n✅ Ma'lumotlar muvaffaqiyatli yuklandi!")
            self.window.append_log("   ➜ Yana CSV qo'shishingiz yoki 'Modelni O'qitish' ni bosishingiz mumkin.")

        except DataValidationError as e:
            self._show_error("Ma'lumot Validatsiyasi Xatosi", str(e))
            self.window.append_log(f"\n❌ XATO: {str(e)}")
            self.window.btn_train.setEnabled(False)

        except InsufficientDiversityError as e:
            self._show_warning("Ma'lumotlar Balansi Ogohlantirishi", str(e))
            self.window.append_log(f"\n⚠️ OGOHLANTIRISH: {str(e)}")
            self.window.btn_train.setEnabled(False)

        except Exception as e:
            self._show_error("Kutilmagan Xato", f"Kutilmagan xato yuz berdi:\n{str(e)}")
            self.window.append_log(f"\n❌ Kutilmagan xato: {str(e)}")
            self.window.btn_train.setEnabled(False)

    def _update_stat_cards(self):
        """Statistika kartochkalarini yangilash."""
        summary = self.data_loader.get_summary()

        self.window.update_stat_card(
            self.window.card_files,
            str(len(summary.get('loaded_files', [])))
        )
        self.window.update_stat_card(
            self.window.card_rows,
            f"{summary.get('total_rows', 0):,}"
        )
        self.window.update_stat_card(
            self.window.card_size,
            format_data_size(summary.get('memory_usage_mb', 0))
        )
        self.window.update_stat_card(
            self.window.card_columns,
            str(summary.get('total_columns', 0))
        )

    def _on_clear_data(self):
        """Barcha yuklangan ma'lumotlarni tozalash."""
        reply = QMessageBox.question(
            self.window,
            "🗑 Tozalash",
            f"Barcha yuklangan ma'lumotlarni tozalashni xohlaysizmi?\n"
            f"({len(self.data_loader.loaded_files)} fayl, "
            f"{len(self.data_loader.dataframe):,} qator)" if self.data_loader.dataframe is not None else "Ma'lumotlarni tozalash?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.data_loader.clear_all()
            self._delete_session()
            self.window.clear_log()
            self.window.set_progress(0)
            self.window.btn_train.setEnabled(False)
            self.window.btn_clear.setEnabled(False)

            # Stat kartochkalarni nolga qaytarish
            self.window.update_stat_card(self.window.card_files, "0")
            self.window.update_stat_card(self.window.card_rows, "0")
            self.window.update_stat_card(self.window.card_size, "0 MB")
            self.window.update_stat_card(self.window.card_columns, "0")

            self.window.statusBar().showMessage("🛡️ IoT-Shield AI Trainer tayyor  |  Ma'lumotlar tozalandi")
            self.window.append_log("🗑 Barcha ma'lumotlar tozalandi.")
            self.window.append_log("📌 Yangi CSV fayllarni yuklang.")

    def _on_train_model(self):
        """Incremental Learning — fayl-fayl o'qitishni boshlash."""
        if not self.data_loader.file_paths:
            self._show_warning(
                "Ma'lumotlar yuklanmagan",
                "Model o'qitish uchun avval CSV fayllarni yuklang."
            )
            return

        if self.data_loader.label_column is None:
            self._show_error(
                "Label ustuni topilmadi",
                "Faylda 'label' ustuni bo'lishi kerak!\n\n"
                "CICIOT2023 datasetida '33 hujum + 1 normal' holat bo'lishi shart."
            )
            return

        # UI ni o'qitish rejimiga o'tkazish
        self.window.set_progress(0)
        self.window.append_log("\n" + "═" * 50)
        self.window.append_log("🧠 INCREMENTAL LEARNING BOSHLANDI")
        self.window.append_log("═" * 50)
        self.window.append_log(f"   Fayllar: {len(self.data_loader.file_paths)} ta")
        self.window.append_log(f"   Features: {len(self.data_loader.feature_columns)} ta")
        self.window.append_log(f"   Label: '{self.data_loader.label_column}'")
        self.window.append_log("")

        self.window.set_training_mode(True)

        # Trainer threadni ishga tushirish
        self.trainer_thread = TrainerThread(data_loader=self.data_loader)
        self.trainer_thread.progress_updated.connect(self.window.set_progress)
        self.trainer_thread.log_message.connect(self.window.append_log)
        self.trainer_thread.training_completed.connect(self._on_training_completed)
        self.trainer_thread.training_failed.connect(self._on_training_failed)
        self.trainer_thread.start()

    def _on_training_completed(self, results: dict):
        """O'qitish yakunlanganda — natijalar va grafiklarni ko'rsatish."""
        self.window.set_training_mode(False)
        self.window.btn_train.setEnabled(True)

        self.window.append_log("\n" + "═" * 50)
        self.window.append_log("✅ INCREMENTAL LEARNING YAKUNLANDI!")
        self.window.append_log("═" * 50)

        status = results.get("status", "")

        if status == "completed":
            acc = results['accuracy'] * 100
            prec = results['precision'] * 100
            rec = results['recall'] * 100
            f1 = results['f1_score'] * 100

            self.window.append_log(f"\n🏆 YAKUNIY NATIJALAR:")
            self.window.append_log(f"   ├── Accuracy:  {acc:.2f}%")
            self.window.append_log(f"   ├── Precision: {prec:.2f}%")
            self.window.append_log(f"   ├── Recall:    {rec:.2f}%")
            self.window.append_log(f"   └── F1-Score:  {f1:.2f}%")

            total_time = results.get('total_time', 0)
            self.window.statusBar().showMessage(
                f"✅ Yakunlandi  |  Accuracy: {acc:.2f}%  |  "
                f"Precision: {prec:.2f}%  |  Recall: {rec:.2f}%  |  "
                f"F1: {f1:.2f}%  |  {results.get('total_files', 0)} fayl  |  {total_time:.0f}s"
            )

            # Grafiklarni chizish
            self._draw_confusion_matrix(results)
            self._draw_accuracy_progression(results)

        self.window.append_log("")

    def _draw_confusion_matrix(self, results: dict):
        """Confusion Matrix heatmap."""
        try:
            cm = results.get('confusion_matrix')
            if cm is None or cm.size == 0:
                return

            class_names = results.get('class_names', [])

            layout = self.window.confusion_placeholder.parent().layout()
            if layout is not None:
                self.window.confusion_placeholder.hide()

            fig = Figure(figsize=(6, 5), dpi=80)
            fig.patch.set_facecolor('#0f172a')
            ax = fig.add_subplot(111)

            # Normalize
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_norm = cm.astype('float') / row_sums
            cm_norm = np.nan_to_num(cm_norm)

            im = ax.imshow(cm_norm, interpolation='nearest', cmap='YlOrRd', aspect='auto')

            ax.set_title('Confusion Matrix', color='white', fontsize=13, fontweight='bold', pad=10)
            ax.set_ylabel('Haqiqiy (True)', color='#94a3b8', fontsize=10)
            ax.set_xlabel('Bashorat (Predicted)', color='#94a3b8', fontsize=10)

            n_classes = cm.shape[0]
            if n_classes <= 15:
                labels = class_names[:n_classes] if len(class_names) >= n_classes else list(range(n_classes))
                ax.set_xticks(range(n_classes))
                ax.set_yticks(range(n_classes))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7, color='#cbd5e1')
                ax.set_yticklabels(labels, fontsize=7, color='#cbd5e1')
            else:
                step = max(1, n_classes // 10)
                ticks = list(range(0, n_classes, step))
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels(ticks, fontsize=8, color='#cbd5e1')
                ax.set_yticklabels(ticks, fontsize=8, color='#cbd5e1')

            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8')

            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelcolor='#94a3b8', labelsize=8)
            fig.tight_layout()

            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(250)
            if layout:
                layout.addWidget(canvas)
            canvas.draw()
            self.window.append_log("\n📊 Confusion Matrix chizildi")

        except Exception as e:
            self.window.append_log(f"\n⚠️ CM xato: {e}")

    def _draw_accuracy_progression(self, results: dict):
        """Fayl bo'yicha accuracy o'sishi grafigini chizish (learning curve)."""
        try:
            per_file_acc = results.get('per_file_accuracy', [])

            layout = self.window.accuracy_placeholder.parent().layout()
            if layout is not None:
                self.window.accuracy_placeholder.hide()

            fig = Figure(figsize=(6, 4), dpi=80)
            fig.patch.set_facecolor('#0f172a')
            ax = fig.add_subplot(111)

            if per_file_acc and len(per_file_acc) > 1:
                # Per-file accuracy progression line
                x = list(range(1, len(per_file_acc) + 1))
                y = [a * 100 for a in per_file_acc]

                ax.plot(x, y, color='#06b6d4', linewidth=2, marker='o',
                        markersize=4, markerfacecolor='#22d3ee', label='Accuracy')
                ax.fill_between(x, y, alpha=0.15, color='#06b6d4')

                ax.set_xlabel('Fayl raqami', color='#94a3b8', fontsize=10)
                ax.set_ylabel('Accuracy (%)', color='#94a3b8', fontsize=10)
                ax.set_title(f'Learning Curve ({len(per_file_acc)} fayl)',
                             color='white', fontsize=13, fontweight='bold', pad=10)

                # Yakuniy accuracy chiziq
                final_acc = per_file_acc[-1] * 100
                ax.axhline(y=final_acc, color='#22c55e', linestyle='--', alpha=0.7, linewidth=1)
                ax.text(len(per_file_acc), final_acc + 1, f'{final_acc:.1f}%',
                        color='#22c55e', fontsize=10, fontweight='bold', ha='right')

            else:
                # Faqat bar chart (kam fayl bo'lsa)
                metrics = {
                    'Accuracy': results.get('accuracy', 0) * 100,
                    'Precision': results.get('precision', 0) * 100,
                    'Recall': results.get('recall', 0) * 100,
                    'F1-Score': results.get('f1_score', 0) * 100,
                }
                names = list(metrics.keys())
                values = list(metrics.values())
                colors = ['#06b6d4', '#22c55e', '#f59e0b', '#a855f7']

                bars = ax.bar(names, values, color=colors, width=0.6)
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f'{val:.1f}%', ha='center', va='bottom',
                            fontsize=11, fontweight='bold', color='white')
                ax.set_ylim(0, 110)
                ax.set_title('Model Metrics', color='white', fontsize=13, fontweight='bold', pad=10)

            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8', labelsize=10)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, color='#334155', linestyle='--', alpha=0.5)

            fig.tight_layout()

            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(200)
            if layout:
                layout.addWidget(canvas)
            canvas.draw()
            self.window.append_log("📊 Learning Curve grafigi chizildi")

        except Exception as e:
            self.window.append_log(f"\n⚠️ Metrics grafigi chizishda xato: {e}")

    def _on_training_failed(self, error_message: str):
        """O'qitishda xato bo'lganda."""
        self.window.set_training_mode(False)
        self.window.btn_train.setEnabled(True)

        self._show_error("O'qitish Xatosi", error_message)
        self.window.append_log(f"\n❌ O'qitish xatosi: {error_message}")

    def _show_error(self, title: str, message: str):
        """Xato message box ko'rsatish."""
        msg = QMessageBox(self.window)
        msg.setWindowTitle(f"❌ {title}")
        msg.setText(message)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
            }}
            QMessageBox QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
                min-width: 400px;
            }}
            QPushButton {{
                background-color: {COLORS['accent_red']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 24px;
                font-weight: 600;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: #f87171;
            }}
        """)
        msg.exec()

    def _show_warning(self, title: str, message: str):
        """Ogohlantirish message box ko'rsatish."""
        msg = QMessageBox(self.window)
        msg.setWindowTitle(f"⚠️ {title}")
        msg.setText(message)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
            }}
            QMessageBox QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
                min-width: 400px;
            }}
            QPushButton {{
                background-color: {COLORS['accent_amber']};
                color: {COLORS['bg_primary']};
                border: none;
                border-radius: 6px;
                padding: 8px 24px;
                font-weight: 600;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: #fbbf24;
            }}
        """)
        msg.exec()

    def _show_about(self):
        """Dastur haqida dialog."""
        msg = QMessageBox(self.window)
        msg.setWindowTitle("🛡️ IoT-Shield AI Trainer haqida")
        msg.setText(
            "IoT-Shield AI Trainer v1.0.0\n\n"
            "CICIOT2023 dataset asosida IoT tarmoq\n"
            "hujumlarini aniqlash tizimi (NIDS).\n\n"
            "Platforma: Raspberry Pi\n"
            "Dataset: CICIOT2023 (39 parametr)\n"
            "Texnologiya: PyQt6 + Python\n\n"
            "© 2026 — Dissertatsiya loyihasi"
        )
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
            }}
            QMessageBox QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {COLORS['accent_cyan']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 24px;
                font-weight: 600;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: #22d3ee;
            }}
        """)
        msg.exec()

    # ═══ SESSION PERSISTENCE ═══

    def _save_session(self, new_paths: list):
        """Yuklangan fayl yo'llarini session.json ga saqlash."""
        try:
            existing = []
            if os.path.exists(SESSION_FILE):
                with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing = data.get('file_paths', [])

            # Yangi yo'llarni qo'shish (dublikatsiz)
            all_paths = list(dict.fromkeys(existing + [os.path.abspath(p) for p in new_paths]))

            with open(SESSION_FILE, 'w', encoding='utf-8') as f:
                json.dump({'file_paths': all_paths}, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.window.append_log(f"\n⚠️ Session saqlashda xato: {e}")

    def _restore_session(self):
        """Avvalgi sessiyani tiklash (agar mavjud bo'lsa)."""
        if not os.path.exists(SESSION_FILE):
            self.window.append_log("📌 CSV fayllarni yuklang yoki papkani tanlang.")
            return

        try:
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                saved_paths = data.get('file_paths', [])

            if not saved_paths:
                self.window.append_log("📌 CSV fayllarni yuklang yoki papkani tanlang.")
                return

            # Mavjud fayllarni filtrlash
            valid_paths = [p for p in saved_paths if os.path.exists(p)]

            if not valid_paths:
                self.window.append_log("⚠️ Avvalgi session fayllari topilmadi.")
                self.window.append_log("📌 Yangi CSV fayllarni yuklang.")
                self._delete_session()
                return

            self.window.append_log(f"🔄 Avvalgi sessiya tiklanmoqda ({len(valid_paths)} fayl)...\n")

            # Ma'lumotlarni yuklash
            self._load_data(valid_paths)

        except Exception as e:
            self.window.append_log(f"⚠️ Session tiklashda xato: {e}")
            self.window.append_log("📌 Yangi CSV fayllarni yuklang.")

    def _delete_session(self):
        """Session faylini o'chirish."""
        try:
            if os.path.exists(SESSION_FILE):
                os.remove(SESSION_FILE)
        except Exception:
            pass

    def run(self):
        """Ilovani ishga tushirish."""
        self.window.show()
        return self.app.exec()


if __name__ == "__main__":
    app = IoTShieldApp()
    sys.exit(app.run())
