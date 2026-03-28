# 🛡️ IoT-Shield — AI-based Network Intrusion Detection System (NIDS)

> **Dissertatsiya loyihasi** — IoT tarmoqlarida AI yordamida hujumlarni aniqlash va bloklash tizimi.
> Dataset: **CICIOT2023** | Platforma: **Raspberry Pi** | Til: **Python**

---

## 📋 Loyiha Haqida Qisqacha

Bu loyiha IoT (Internet of Things) tarmoqlarini real vaqtda kuzatib, sun'iy intellekt yordamida tarmoq hujumlarini aniqlash va avtomatik bloklash tizimi. Tizim ikki asosiy qismdan iborat:

1. **AI Trainer (Desktop)** — CICIOT2023 datasetidan model o'qitish uchun PyQt6 GUI dastur
2. **Guard (Raspberry Pi)** — O'qitilgan modelni ishlatib, tarmoqni real vaqtda himoyalash + Telegram Bot

---

## 🏗️ Arxitektura va Fayl Tuzilmasi

```
AI Project/
├── main.py            # AI Trainer — asosiy GUI kontroller
├── ui_design.py       # PyQt6 UI dizayni (Dark Theme Dashboard)
├── data_loader.py     # CICIOT2023 dataset yuklash va preprocessing
├── trainer.py         # SGDClassifier — Incremental Learning engine
├── detector.py        # Real-time paket sniffing va AI bashorat
├── bot_handler.py     # Telegram Bot (aiogram 3.x)
├── guard.py           # Entry point — Bot + Detector birga ishga tushirish
├── requirements.txt   # Python kutubxonalari
├── session.json       # Oxirgi sessiya ma'lumotlari (auto-generated)
├── models/            # O'qitilgan model fayllari (.pkl + .json)
│   ├── iot_shield_sgd_*.pkl       # SGDClassifier model
│   ├── iot_shield_rf_*.pkl        # RandomForest model (eski)
│   ├── scaler_*.pkl               # StandardScaler
│   ├── label_encoder_*.pkl        # LabelEncoder
│   └── metadata_*.json            # Model metadata
└── logs/              # Log fayllar
    ├── guard.log      # Guard ishga tushgandagi loglar
    ├── detector.log   # Detector mustaqil rejim loglari
    └── debug_log.txt  # Feature debug loglari (FP tahlili uchun)
```

---

## 🧠 Ishlatilgan Algoritmlar va Modellar

### 1. **SGDClassifier (Stochastic Gradient Descent) — Asosiy Model**

```
Model:         SGDClassifier (sklearn.linear_model)
Loss:          modified_huber (ehtimollik beradi, ko'p klassli)
Penalty:       L2 regularization
Alpha:         1e-4
O'qitish:      Incremental Learning (partial_fit)
```

**Nima uchun SGDClassifier tanlandi:**
- **Incremental Learning** (partial_fit) — katta datasetlarni RAM ga sig'maydigan holatda bo'laklab o'qitish mumkin
- **Modified Huber loss** — predict_proba() orqali ishonch darajasini (confidence) olish imkonini beradi
- **Raspberry Pi uchun mos** — engil model, tezkor predict qiladi
- **L2 regularization** — overfitting oldini oladi

### 2. **RandomForestClassifier — Dastlabki Model (eski)**

Loyihaning birinchi bosqichida `RandomForestClassifier` ishlatilgan edi. Lekin u:
- Katta modellar hosil qiladi (~32 MB .pkl fayl)
- partial_fit qo'llab-quvvatlamaydi (bir martalik o'qitish)
- Raspberry Pi uchun og'ir

Shuning uchun SGDClassifier ga o'tildi.

---

## 📊 Dataset: CICIOT2023

### Ma'lumotlar tuzilishi:
- **39 ta feature** (parametr)
- **34 ta klass** (33 hujum turi + 1 normal/BENIGN)
- Fayllar CSV formatda, jami hajmi ~20-40 GB

### O'qitishda ishlatiladigan 37 ta feature (detector.py da):

```python
FEATURE_ORDER = [
    "ARP", "AVG", "DHCP", "DNS", "HTTP", "HTTPS", "Header_Length",
    "IAT", "ICMP", "IPv", "IRC", "LLC", "Max", "Min", "Number",
    "Protocol Type", "Rate", "SMTP", "SSH", "Std", "TCP", "Telnet",
    "Tot size", "Tot sum", "UDP", "Variance", "ack_count",
    "ack_flag_number", "cwr_flag_number", "ece_flag_number",
    "fin_count", "fin_flag_number", "psh_flag_number", "rst_count",
    "rst_flag_number", "syn_count", "syn_flag_number",
]
```

### Data Pipeline (Data Loader logikasi):
1. CSV fayllar yuklanadi → faqat birinchi 5000 qator preview uchun o'qiladi
2. **auto_detect_columns()** — CICIOT2023 39 ta feature va label ustunini avtomatik aniqlaydi
3. **scan_all_classes()** — barcha fayllardan noyob klasslarni skanlab, LabelEncoder yaratadi
4. **fit_scaler_from_first_file()** — StandardScaler ni birinchi fayldan partial_fit qiladi
5. **stream_file_chunks()** — generator orqali 100,000 qatorlik chunk'lar bilan streaming
6. Har bir chunk: NaN/Inf → 0, float64 → float32 downcast, scaler.transform(), label encode

---

## 🔄 Training Pipeline (O'qitish Jarayoni)

`trainer.py` dagi TrainerThread QThread orqali alohida threadda ishlaydi:

```
1-BOSQICH: Klasslarni aniqlash
    └── Barcha fayllardan label ustunini o'qib, noyob klasslarni topish
    └── LabelEncoder.fit(sorted_class_names)

2-BOSQICH: Scaler o'rganish
    └── StandardScaler.partial_fit(birinchi_fayl)

3-BOSQICH: Model yaratish va Incremental O'qitish
    └── SGDClassifier(loss='modified_huber', penalty='l2')
    └── Har bir fayl chunk-lab o'qiladi (100K qator)
    └── Har bir chunk shuffle qilinadi (balans uchun)
    └── model.partial_fit(X_chunk, y_chunk, classes=all_classes)
    └── Oxirgi faylning 20% si test uchun ajratiladi

4-BOSQICH: Model baholash
    └── accuracy_score, precision_score, recall_score, f1_score
    └── confusion_matrix, classification_report
    └── Per-file accuracy progression (learning curve)

5-BOSQICH: Model saqlash
    └── model.pkl (joblib, compress=3)
    └── scaler.pkl
    └── label_encoder.pkl
    └── metadata.json (feature_names, class_names, n_features, n_classes, timestamps)
```

### Baholash metrikalari:
- **Accuracy** — umumiy to'g'rilik
- **Precision** (weighted) — qaysi darajada to'g'ri topilgan
- **Recall** (weighted) — qaysi darajada barcha hujumlar topilgan
- **F1-Score** (weighted) — Precision va Recall balansi
- **Confusion Matrix** — heatmap grafik (matplotlib)
- **Learning Curve** — fayl bo'yicha accuracy o'sishi

---

## 🔍 Detection Engine (Real-time Aniqlash Mantiqiy)

`detector.py` dagi `DetectionEngine` v2.0 quyidagi mantiqda ishlaydi:

### Paket Yig'ish (FlowWindow):
```
1 soniyalik sliding window
├── Har bir paketdan: IP, port, protokol, hajm, TCP flaglar
├── 37 ta CICIOT2023 feature hisoblash
└── Scapy kutubxonasi orqali sniffing
```

### Qaror Darajalari (False Positive Prevention):
```
┌─────────────────────────────────────────────────────────────────┐
│ 1. IP SAFE LIST'da        → O'tkazib yuborish (hech narsa)     │
│ 2. Model: BENIGN          → Normal trafik (log)                │
│ 3. Ishonch < 70% YOKI     → FP_PREVENT (debug logga yozish)   │
│    Paketlar < 5                                                │
│ 4. 70% ≤ Ishonch < 95%    → ALERT_ONLY (Telegram xabar,       │
│    YOKI Paketlar < 10        bloklamasdan)                     │
│ 5. Ishonch ≥ 95% VA       → BLOCK (IP bloklash + Alert)       │
│    Paketlar ≥ 10                                               │
│ 6. Bir IP 3+ marta        → REPEAT BLOCK (ishonchdan           │
│    ketma-ket hujum           qat'i nazar bloklash)             │
└─────────────────────────────────────────────────────────────────┘
```

### Konfiguratsiya konstantalari:
```python
BLOCK_THRESHOLD     = 0.95     # 95% dan past — bloklanmaydi
ALERT_THRESHOLD     = 0.70     # 70% dan yuqori — Telegram alert
MIN_PACKETS_TO_BLOCK = 10      # Kamida 10 paket bo'lmasa bloklanmaydi
MIN_PACKETS_TO_ALERT = 5       # Kamida 5 paket kerak alertga ham
REPEAT_ATTACK_COUNT  = 3       # 3 marta ketma-ket → bloklash
MAX_BLOCKED_IPS      = 500     # Maksimum bloklangan IP soni
```

### Safe List (Whitelist):
- Avtomatik: loopback, broadcast, default gateway, DNS serverlar, local IP
- `_detect_gateways()` — ipconfig/ip route orqali gateway topish
- `_detect_dns_servers()` — Google DNS, Cloudflare, Quad9, va lokal DNS
- `_detect_local_ip()` — socket orqali lokal IP aniqlash
- `add_active_connections()` — faol ulanishlarni safe listga qo'shish

### Debug Feature Logger:
- Har bir bashoratning feature qiymatlarini `logs/debug_log.txt` ga yozib boradi
- False Positive sababini aniqlash uchun muhim
- Maksimum 10,000 qator

---

## 🤖 Telegram Bot (aiogram 3.x)

`bot_handler.py` dagi `IoTShieldBot` quyidagi komandalar bilan ishlaydi:

| Komanda | Tavsif |
|---------|--------|
| `/start` | Bot haqida ma'lumot, Chat ID avtomatik saqlash |
| `/scan` | ARP orqali tarmoqdagi qurilmalarni topish |
| `/scan 192.168.0.0/16` | Maxsus diapazonni skan qilish |
| `/block IP` | IP manzilni firewall orqali bloklash |
| `/unblock IP` | IP ni blokdan ochish |
| `/blocked` | Bloklangan IP lar ro'yxati |
| `/safelist` | Xavfsiz IP lar ro'yxati |
| `/safelist add IP` | Safe Listga IP qo'shish |
| `/status` | Tizim holati va statistikalar |
| `/help` | Yordam |

### Bot Xususiyatlari:
- **Hujum alertlari** — hujum aniqlanganda avtomatik Telegram xabar + Inline "Blokdan ochish" tugmasi
- **Adaptive ARP Scan** — barcha subnetlarni avtomatik aniqlash va skan qilish
- **MAC OUI Lookup** — 60+ ta ishlab chiqaruvchi (Raspberry Pi, ESP32, Xiaomi, TP-Link, ...)
- **Cross-platform Firewall** — Windows (netsh) va Linux (iptables) uchun IP bloklash
- **Startup xabar** — bot ishga tushganda bildirishnoma yuborish

---

## 🖥️ GUI Dashboard (PyQt6)

`ui_design.py` — Premium dark theme cybersecurity dashboard:

### Dizayn:
- **Rang palitrasi**: Dark mode (#0a0e17 asosiy fon, cyan/green/purple aksentlar)
- **Shriftlar**: Segoe UI (UI), Cascadia Code/Fira Code (log terminal)
- **Gradient tugmalar**, hover effektlar, borderRadius: 8-12px

### UI Komponentlari:
1. **Header** — Logo, sarlavha, versiya, "RPi Ready" ko'rsatkich
2. **Stat Cards** (4 ta) — Yuklangan fayllar, Jami qatorlar, Ma'lumot hajmi, Ustunlar
3. **Action Buttons** — CSV qo'shish, Papka qo'shish, Tozalash, Modelni O'qitish
4. **Progress Bar** — gradient to'ldirish animatsiyasi
5. **Real-vaqtli Log** — monospace terminal
6. **Grafiklar paneli** — Confusion Matrix heatmap + Learning Curve / Metrics bar chart

### Session Management:
- `session.json` — yuklangan CSV fayl yo'llarini saqlaydi
- Ilovani qayta ochganda avtomatik tiklaydi

---

## 🚀 Ishga Tushirish

### 1. O'rnatish:
```bash
pip install -r requirements.txt
```

### 2. AI Trainer (Model O'qitish — Desktop):
```bash
python main.py
```
- PyQt6 dashboard ochiladi
- CSV fayllarni yuklang → "Modelni O'qitish" tugmasini bosing
- Model `models/` papkasiga saqlanadi

### 3. Guard (Real-time Himoya — Raspberry Pi yoki Desktop):
```bash
# Bot + Detector birga
sudo python3 guard.py --interface eth0

# Faqat bot (test uchun)
python guard.py --bot-only

# Maxsus parametrlar bilan
sudo python3 guard.py -i wlan0 -t 0.7 --chat-id 123456789
```

### 4. Detector mustaqil (bot siz):
```bash
sudo python3 detector.py -i eth0 --block-threshold 0.95 --alert-threshold 0.70
```

---

## 📦 Texnologiya Steki

| Qatlam | Texnologiya | Versiya |
|--------|-------------|---------|
| ML Model | scikit-learn (SGDClassifier) | >= 1.3.0 |
| Data Processing | pandas, numpy | >= 2.0, >= 1.24 |
| Model Serialization | joblib | >= 1.3.0 |
| GUI | PyQt6 | >= 6.5.0 |
| Grafiklar | matplotlib | >= 3.7.0 |
| Telegram Bot | aiogram | >= 3.4.0 |
| Network Sniffing | scapy | >= 2.5.0 |
| Feature Scaling | StandardScaler | (sklearn) |
| Label Encoding | LabelEncoder | (sklearn) |

---

## 📝 Qilingan Ishlar Tarixi

### v1.0 — Dastlabki versiya (2026-03-24)
- CICIOT2023 dataset uchun data loader yaratildi (39 ta feature)
- RandomForestClassifier bilan model o'qitish
- PyQt6 dashboard (dark theme)
- Confusion Matrix va Metrics grafiklar

### v1.5 — Incremental Learning (2026-03-24)
- RandomForest → SGDClassifier (partial_fit) ga o'tildi
- Chunk-based streaming data pipeline (100K qator)
- float64 → float32 memory optimization
- Per-file accuracy progression (learning curve)

### v2.0 — Guard + False Positive Prevention (2026-03-26)
- `guard.py` — Telegram Bot + Detector entry point
- `detector.py` — Real-time AI Detection Engine v2.0
- `bot_handler.py` — Telegram Bot (aiogram 3.x, 8 ta komanda)
- **False Positive Prevention:**
  - 95% threshold + minimum 10 paket bloklash uchun
  - 70% alert threshold (bloklamasdan xabar)
  - Repeat attack counter (3+ marta → bloklash)
- **Safe List (Whitelist):**
  - Gateway, DNS, lokal IP avtomatik aniqlash
  - Telegram orqali boshqarish (`/safelist add/remove`)
  - Faol ulanishlarni avtomatik qo'shish
- **Adaptive ARP Scan** — bir necha subnet skan
- **MAC OUI Database** — 60+ ishlab chiqaruvchi aniqlash
- **Cross-platform Firewall** — Windows (netsh) + Linux (iptables)
- **Debug Feature Logger** — `debug_log.txt` (FP tahlili uchun)

---

## ⚠️ Muhim Eslatmalar

1. **Model fayllari** `models/` papkasida saqlanadi. Eng oxirgi model avtomatik tanlanadi (metadata fayl nomi bo'yicha saralash).
2. **Guard** ishga tushirish uchun **admin/root** huquqi kerak (IP bloklash, paket sniffing).
3. **Telegram Bot Token** `bot_handler.py` da hardcoded. Ishlab chiqarishda environment variable ga o'tkazish kerak.
4. **CICIOT2023 dataset** juda katta (~20-40 GB). Incremental Learning buning ustidan ishlaydi — hamma narsani RAM ga yuklamaydi.
5. **Raspberry Pi** da ishlatish uchun model `.pkl` fayllarini Pi ga nusxalash kerak.

---

## 🔗 Foydali Havolalar

- CICIOT2023 Dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- scikit-learn SGDClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
- aiogram 3.x: https://docs.aiogram.dev/en/latest/
- scapy: https://scapy.readthedocs.io/

---

*Oxirgi yangilanish: 2026-03-28*
