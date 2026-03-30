"""
IoT-Shield AI Trainer - UI Design Module
PyQt6 bilan premium dizaynli Cybersecurity NIDS Dashboard.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QTextEdit, QLabel,
    QFrame, QSplitter, QGroupBox, QSizePolicy,
    QStatusBar, QMenuBar
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette, QAction

# ─── Rang palitrasi ───────────────────────────────────────────────
COLORS = {
    "bg_primary": "#0a0e17",
    "bg_secondary": "#111827",
    "bg_card": "#1a2332",
    "bg_card_hover": "#1f2b3d",
    "accent_cyan": "#06b6d4",
    "accent_green": "#10b981",
    "accent_red": "#ef4444",
    "accent_amber": "#f59e0b",
    "accent_purple": "#8b5cf6",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "border": "#1e293b",
    "border_active": "#06b6d4",
    "progress_bg": "#1e293b",
    "progress_fill": "#06b6d4",
    "log_bg": "#0d1117",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
}


def get_stylesheet() -> str:
    """Asosiy dark-theme stylesheet qaytarish."""
    return f"""
    /* ═══ GLOBAL ═══ */
    QMainWindow {{
        background-color: {COLORS['bg_primary']};
    }}

    QWidget {{
        color: {COLORS['text_primary']};
        font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
    }}

    /* ═══ MENU BAR ═══ */
    QMenuBar {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_secondary']};
        border-bottom: 1px solid {COLORS['border']};
        padding: 4px;
        font-size: 13px;
    }}
    QMenuBar::item:selected {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['accent_cyan']};
        border-radius: 4px;
    }}
    QMenu {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 4px;
    }}
    QMenu::item:selected {{
        background-color: {COLORS['bg_card_hover']};
        color: {COLORS['accent_cyan']};
    }}

    /* ═══ STATUS BAR ═══ */
    QStatusBar {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_muted']};
        border-top: 1px solid {COLORS['border']};
        font-size: 12px;
        padding: 2px 8px;
    }}

    /* ═══ LABELS ═══ */
    QLabel {{
        color: {COLORS['text_primary']};
        font-size: 13px;
    }}

    /* ═══ BUTTONS ═══ */
    QPushButton {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 14px;
        font-weight: 600;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['bg_card_hover']};
        border-color: {COLORS['accent_cyan']};
        color: {COLORS['accent_cyan']};
    }}
    QPushButton:pressed {{
        background-color: {COLORS['accent_cyan']};
        color: {COLORS['bg_primary']};
    }}
    QPushButton:disabled {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_muted']};
        border-color: {COLORS['border']};
    }}

    /* ═══ PRIMARY BUTTON ═══ */
    QPushButton#btn_load_csv {{
        background: qlineargradient(
            spread:pad, x1:0, y1:0, x2:1, y2:0,
            stop:0 #06b6d4, stop:1 #8b5cf6
        );
        color: white;
        border: none;
        font-size: 15px;
    }}
    QPushButton#btn_load_csv:hover {{
        background: qlineargradient(
            spread:pad, x1:0, y1:0, x2:1, y2:0,
            stop:0 #22d3ee, stop:1 #a78bfa
        );
    }}
    QPushButton#btn_load_csv:disabled {{
        background: {COLORS['bg_secondary']};
        color: {COLORS['text_muted']};
    }}

    QPushButton#btn_train {{
        background: qlineargradient(
            spread:pad, x1:0, y1:0, x2:1, y2:0,
            stop:0 #10b981, stop:1 #06b6d4
        );
        color: white;
        border: none;
        font-size: 15px;
    }}
    QPushButton#btn_train:hover {{
        background: qlineargradient(
            spread:pad, x1:0, y1:0, x2:1, y2:0,
            stop:0 #34d399, stop:1 #22d3ee
        );
    }}
    QPushButton#btn_train:disabled {{
        background: {COLORS['bg_secondary']};
        color: {COLORS['text_muted']};
    }}

    QPushButton#btn_load_dir {{
        background: qlineargradient(
            spread:pad, x1:0, y1:0, x2:1, y2:0,
            stop:0 #f59e0b, stop:1 #ef4444
        );
        color: white;
        border: none;
        font-size: 14px;
    }}
    QPushButton#btn_load_dir:hover {{
        background: qlineargradient(
            spread:pad, x1:0, y1:0, x2:1, y2:0,
            stop:0 #fbbf24, stop:1 #f87171
        );
    }}
    QPushButton#btn_load_dir:disabled {{
        background: {COLORS['bg_secondary']};
        color: {COLORS['text_muted']};
    }}

    QPushButton#btn_clear {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['accent_red']};
        border: 1px solid {COLORS['accent_red']};
        font-size: 13px;
    }}
    QPushButton#btn_clear:hover {{
        background-color: {COLORS['accent_red']};
        color: white;
    }}
    QPushButton#btn_clear:disabled {{
        background: {COLORS['bg_secondary']};
        color: {COLORS['text_muted']};
        border-color: {COLORS['border']};
    }}

    /* ═══ PROGRESS BAR ═══ */
    QProgressBar {{
        background-color: {COLORS['progress_bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        text-align: center;
        color: {COLORS['text_primary']};
        font-size: 12px;
        font-weight: bold;
        min-height: 22px;
        max-height: 22px;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(
            spread:pad, x1:0, y1:0, x2:1, y2:0,
            stop:0 #06b6d4, stop:0.5 #8b5cf6, stop:1 #10b981
        );
        border-radius: 9px;
    }}

    /* ═══ LOG TEXT EDIT ═══ */
    QTextEdit {{
        background-color: {COLORS['log_bg']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 12px;
        font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
        font-size: 12px;
        selection-background-color: {COLORS['accent_cyan']};
        selection-color: {COLORS['bg_primary']};
    }}

    /* ═══ GROUP BOX ═══ */
    QGroupBox {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        margin-top: 16px;
        padding: 20px 16px 16px 16px;
        font-size: 14px;
        font-weight: 600;
        color: {COLORS['accent_cyan']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 4px 16px;
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['accent_cyan']};
    }}

    /* ═══ FRAME ═══ */
    QFrame#card_frame {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 16px;
    }}

    /* ═══ SPLITTER ═══ */
    QSplitter::handle {{
        background-color: {COLORS['border']};
        width: 2px;
        margin: 4px;
    }}
    QSplitter::handle:hover {{
        background-color: {COLORS['accent_cyan']};
    }}
    """


def create_stat_card(title: str, value: str, color: str = COLORS['accent_cyan']) -> QFrame:
    """Statistika kartochkasi yaratish."""
    frame = QFrame()
    frame.setObjectName("card_frame")
    frame.setFixedHeight(90)
    frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    layout = QVBoxLayout(frame)
    layout.setSpacing(0)
    layout.setContentsMargins(14, 6, 14, 6)

    title_label = QLabel(title)
    title_label.setFixedHeight(18)
    title_label.setStyleSheet(f"""
        color: {COLORS['text_secondary']};
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    """)

    value_label = QLabel(value)
    value_label.setObjectName(f"stat_{title.lower().replace(' ', '_')}")
    value_label.setMinimumHeight(50)
    value_label.setStyleSheet(f"""
        color: {color};
        font-size: 32px;
        font-weight: 800;
        padding-top: 0px;
    """)

    layout.addWidget(title_label)
    layout.addWidget(value_label, 1)

    return frame


class MainWindow(QMainWindow):
    """IoT-Shield AI Trainer asosiy oynasi."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛡️ IoT-Shield AI Trainer — CICIOT2023 NIDS Dashboard")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Stylesheet o'rnatish
        self.setStyleSheet(get_stylesheet())

        # UI yaratish
        self._setup_menubar()
        self._setup_ui()
        self._setup_statusbar()

    def _setup_menubar(self):
        """Menu bar yaratish."""
        menubar = self.menuBar()

        # Fayl menyu
        file_menu = menubar.addMenu("📁 &Fayl")

        self.action_open_csv = QAction("CSV Fayllarni yuklash", self)
        self.action_open_csv.setShortcut("Ctrl+O")
        file_menu.addAction(self.action_open_csv)

        self.action_open_dir = QAction("MERGED_CSV papkasini yuklash", self)
        self.action_open_dir.setShortcut("Ctrl+D")
        file_menu.addAction(self.action_open_dir)

        file_menu.addSeparator()

        self.action_exit = QAction("Chiqish", self)
        self.action_exit.setShortcut("Ctrl+Q")
        file_menu.addAction(self.action_exit)

        # Yordam menyu
        help_menu = menubar.addMenu("❓ &Yordam")

        self.action_about = QAction("Dastur haqida", self)
        help_menu.addAction(self.action_about)

    def _setup_statusbar(self):
        """Status bar yaratish."""
        self.statusBar().showMessage("🛡️ IoT-Shield AI Trainer tayyor  |  Ma'lumotlar yuklanmagan")

    def _setup_ui(self):
        """Asosiy UI komponentlarini yaratish."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(16, 8, 16, 6)

        # ═══ HEADER ═══
        header = self._create_header()
        main_layout.addWidget(header)

        # ═══ STAT CARDS ═══
        stats_layout = self._create_stat_cards()
        main_layout.addLayout(stats_layout)

        # ═══ ACTION BUTTONS ═══
        buttons_layout = self._create_action_buttons()
        main_layout.addLayout(buttons_layout)

        # ═══ PROGRESS BAR ═══
        progress_section = self._create_progress_section()
        main_layout.addWidget(progress_section)

        # ═══ MAIN CONTENT (Log + Grafik) ═══
        content_splitter = self._create_content_area()
        main_layout.addWidget(content_splitter, 1)

    def _create_header(self) -> QFrame:
        """Dashboard sarlavha qismi."""
        header_frame = QFrame()
        header_frame.setMaximumHeight(50)
        header_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['bg_card']}, stop:1 {COLORS['bg_secondary']}
                );
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 2px;
            }}
        """)

        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(16, 4, 16, 4)

        # Logo va sarlavha — bitta qatorda
        title = QLabel("🛡️ IoT-Shield AI Trainer")
        title.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 800;
            color: {COLORS['text_primary']};
            background: transparent;
            border: none;
        """)

        separator = QLabel("  ·  ")
        separator.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 14px;
            background: transparent;
            border: none;
        """)

        subtitle = QLabel("CICIOT2023 — NIDS Dashboard")
        subtitle.setStyleSheet(f"""
            font-size: 12px;
            color: {COLORS['text_muted']};
            background: transparent;
            border: none;
        """)

        # O'ng qism
        version_label = QLabel("v1.0.0")
        version_label.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 11px;
            background: {COLORS['bg_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 2px 8px;
        """)

        platform_label = QLabel("🍓 RPi Ready")
        platform_label.setStyleSheet(f"""
            color: {COLORS['accent_green']};
            font-size: 11px;
            background: transparent;
            border: none;
        """)

        header_layout.addWidget(title)
        header_layout.addWidget(separator)
        header_layout.addWidget(subtitle)
        header_layout.addStretch(1)
        header_layout.addWidget(platform_label)
        header_layout.addWidget(version_label)

        return header_frame

    def _create_stat_cards(self) -> QHBoxLayout:
        """Statistika kartochkalari."""
        layout = QHBoxLayout()
        layout.setSpacing(12)

        self.card_files = create_stat_card("Yuklangan fayllar", "0", COLORS['accent_cyan'])
        self.card_rows = create_stat_card("Jami qatorlar", "0", COLORS['accent_green'])
        self.card_size = create_stat_card("Ma'lumot hajmi", "0 MB", COLORS['accent_purple'])
        self.card_columns = create_stat_card("Ustunlar", "0", COLORS['accent_amber'])

        layout.addWidget(self.card_files)
        layout.addWidget(self.card_rows)
        layout.addWidget(self.card_size)
        layout.addWidget(self.card_columns)

        return layout

    def _create_action_buttons(self) -> QHBoxLayout:
        """Amal tugmalari."""
        layout = QHBoxLayout()
        layout.setSpacing(8)

        # CSV fayllarni yuklash tugmasi
        self.btn_load_csv = QPushButton("📂  CSV Qo'shish")
        self.btn_load_csv.setObjectName("btn_load_csv")
        self.btn_load_csv.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_csv.setMinimumHeight(38)
        self.btn_load_csv.setMaximumHeight(42)

        # MERGED_CSV papkasini yuklash
        self.btn_load_dir = QPushButton("📁  Papkadan Qo'shish")
        self.btn_load_dir.setObjectName("btn_load_dir")
        self.btn_load_dir.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_dir.setMinimumHeight(38)
        self.btn_load_dir.setMaximumHeight(42)

        # Tozalash tugmasi
        self.btn_clear = QPushButton("🗑  Tozalash")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setMinimumHeight(38)
        self.btn_clear.setMaximumHeight(42)
        self.btn_clear.setEnabled(False)

        # Modelni o'qitish tugmasi
        self.btn_train = QPushButton("🧠  Modelni O'qitish")
        self.btn_train.setObjectName("btn_train")
        self.btn_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_train.setMinimumHeight(38)
        self.btn_train.setMaximumHeight(42)
        self.btn_train.setEnabled(False)

        layout.addWidget(self.btn_load_csv, 3)
        layout.addWidget(self.btn_load_dir, 3)
        layout.addWidget(self.btn_clear, 1)
        layout.addWidget(self.btn_train, 3)

        return layout

    def _create_progress_section(self) -> QFrame:
        """Progress bar qismi."""
        frame = QFrame()
        frame.setObjectName("card_frame")
        frame.setMaximumHeight(58)

        layout = QVBoxLayout(frame)
        layout.setSpacing(6)
        layout.setContentsMargins(16, 10, 16, 10)

        # Progress label
        prog_header = QHBoxLayout()
        self.progress_label = QLabel("⏳ Kutilmoqda...")
        self.progress_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
            font-weight: 500;
        """)

        self.progress_percent = QLabel("0%")
        self.progress_percent.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-size: 12px;
            font-weight: 700;
        """)
        self.progress_percent.setAlignment(Qt.AlignmentFlag.AlignRight)

        prog_header.addWidget(self.progress_label)
        prog_header.addWidget(self.progress_percent)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        layout.addLayout(prog_header)
        layout.addWidget(self.progress_bar)

        return frame

    def _create_content_area(self) -> QSplitter:
        """Asosiy kontent: Log oynasi va Grafik joylari."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # ═══ LEFT: Log oynasi ═══
        log_group = QGroupBox("📋 Real-vaqtli Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 12, 8, 8)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText(
            "Bu yerda o'qitish jarayoni haqida real-vaqtli ma'lumotlar ko'rsatiladi...\n\n"
            "Boshlash uchun CSV fayllarni yuklang ➜ Modelni o'qiting"
        )

        log_layout.addWidget(self.log_text)

        # ═══ RIGHT: Grafik joylari ═══
        graph_group = QGroupBox("📊 Natijalar va Grafiklar")
        graph_layout = QVBoxLayout(graph_group)
        graph_layout.setContentsMargins(8, 12, 8, 8)
        graph_layout.setSpacing(12)

        # Confusion Matrix uchun joy
        self.confusion_frame = QFrame()
        self.confusion_frame.setObjectName("card_frame")
        self.confusion_frame.setMinimumHeight(200)
        confusion_layout = QVBoxLayout(self.confusion_frame)

        confusion_title = QLabel("🔢 Confusion Matrix")
        confusion_title.setStyleSheet(f"""
            color: {COLORS['accent_purple']};
            font-size: 14px;
            font-weight: 600;
        """)
        confusion_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.confusion_placeholder = QLabel(
            "📊 Model o'qitilgandan so'ng\nConfusion Matrix shu yerda ko'rsatiladi"
        )
        self.confusion_placeholder.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 12px;
        """)
        self.confusion_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)

        confusion_layout.addWidget(confusion_title)
        confusion_layout.addWidget(self.confusion_placeholder, 1)

        # Accuracy grafik uchun joy
        self.accuracy_frame = QFrame()
        self.accuracy_frame.setObjectName("card_frame")
        self.accuracy_frame.setMinimumHeight(200)
        accuracy_layout = QVBoxLayout(self.accuracy_frame)

        accuracy_title = QLabel("📈 Accuracy & Metrics")
        accuracy_title.setStyleSheet(f"""
            color: {COLORS['accent_green']};
            font-size: 14px;
            font-weight: 600;
        """)
        accuracy_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.accuracy_placeholder = QLabel(
            "📈 Model o'qitilgandan so'ng\nAccuracy grafigi shu yerda ko'rsatiladi"
        )
        self.accuracy_placeholder.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 12px;
        """)
        self.accuracy_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)

        accuracy_layout.addWidget(accuracy_title)
        accuracy_layout.addWidget(self.accuracy_placeholder, 1)

        graph_layout.addWidget(self.confusion_frame, 1)
        graph_layout.addWidget(self.accuracy_frame, 1)

        # Splitterga qo'shish
        splitter.addWidget(log_group)
        splitter.addWidget(graph_group)
        splitter.setSizes([650, 350])

        return splitter

    def update_stat_card(self, card_frame: QFrame, value: str):
        """Statistika kartochkasidagi qiymatni yangilash."""
        value_labels = card_frame.findChildren(QLabel)
        if len(value_labels) >= 2:
            value_labels[1].setText(value)

    def append_log(self, message: str):
        """Log oynasiga xabar qo'shish."""
        self.log_text.append(message)
        # Avtomatik pastga scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """Log oynasini tozalash."""
        self.log_text.clear()

    def set_progress(self, value: int):
        """Progress banni yangilash."""
        self.progress_bar.setValue(value)
        self.progress_percent.setText(f"{value}%")

        if value == 0:
            self.progress_label.setText("⏳ Kutilmoqda...")
        elif value < 100:
            self.progress_label.setText("🔄 O'qitish jarayonida...")
        else:
            self.progress_label.setText("✅ Yakunlandi!")

    def set_training_mode(self, training: bool):
        """O'qitish rejimini o'rnatish (tugmalarni enable/disable)."""
        self.btn_load_csv.setEnabled(not training)
        self.btn_load_dir.setEnabled(not training)
        self.btn_clear.setEnabled(not training)
        self.btn_train.setEnabled(not training)
