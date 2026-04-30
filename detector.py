"""
IoT-Shield Guard — AI Detection Engine v2.0
=============================================
O'qitilgan SGDClassifier modelini ishlatib, tarmoq trafigini
real vaqtda kuzatadi va bot_handler orqali xabar beradi.

v2.0 yangiliklari:
  - False Positive Prevention (95% threshold + min_packets)
  - Whitelist / Safe List (gateway, DNS, tez-tez ishlatiladigan IP)
  - Debug Feature Logger (debug_log.txt)
  - Yaxshilangan statistika & tahlil

Bu modul guard.py orqali ishga tushiriladi.
Mustaqil ham ishlaydi: sudo python3 detector.py
"""

import os
import sys
import json
import time
import logging
import asyncio
import argparse
import socket
from collections import defaultdict, deque
from datetime import datetime
from threading import Thread, Lock
from typing import Optional, Set

import numpy as np
import joblib

from feature_engineering import add_derived_features_np, DERIVED_FEATURES, FEATURE_ENG_VERSION
from data_loader import CLASS_GROUPING

try:
    from scapy.all import (
        sniff, IP, TCP, UDP, ICMP, ARP, Ether, Raw, conf
    )
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False
    print("⚠️  scapy kutubxonasi topilmadi: pip install scapy")

# ═══════════════════════════════════════════════════════════
# KONFIGURATSIYA
# ═══════════════════════════════════════════════════════════

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
DEBUG_LOG_FILE = os.path.join(LOG_DIR, "debug_log.txt")

WINDOW_SECONDS = 1.0

# ═══ FALSE POSITIVE PREVENTION ═══
BLOCK_THRESHOLD = 0.95        # 95% dan past — BLOKLANMAYDI, faqat log
ALERT_THRESHOLD = 0.70        # 70% dan yuqori — Telegram alert (bloklamasdan)
MIN_PACKETS_TO_BLOCK = 10     # Kamida 10 paket bo'lmasa bloklanmaydi
MIN_PACKETS_TO_ALERT = 5      # Kamida 5 paket bo'lmasa alert ham yuborilmaydi
REPEAT_ATTACK_COUNT = 3       # Bir IP 3 marta ketma-ket hujum qilsa bloklash

MAX_BLOCKED_IPS = 500
BLOCK_COOLDOWN = 300

# CICIOT2023 — 37 ta feature (model metadata tartibida)
FEATURE_ORDER = [
    "ARP", "AVG", "DHCP", "DNS", "HTTP", "HTTPS", "Header_Length",
    "IAT", "ICMP", "IPv", "IRC", "LLC", "Max", "Min", "Number",
    "Protocol Type", "Rate", "SMTP", "SSH", "Std", "TCP", "Telnet",
    "Tot size", "Tot sum", "UDP", "Variance", "ack_count",
    "ack_flag_number", "cwr_flag_number", "ece_flag_number",
    "fin_count", "fin_flag_number", "psh_flag_number", "rst_count",
    "rst_flag_number", "syn_count", "syn_flag_number",
]

BENIGN_LABELS = {"BENIGN", "Normal", "normal", "benign", "BenignTraffic"}

# Yangi 16-sinf class grouping tizimidagi hamma guruh nomlari
# (detector metadata dan olinadigan class_names bilan moslashtirish uchun)
GROUPED_CLASS_NAMES = set(CLASS_GROUPING.values())

logger = logging.getLogger("IoT-Shield.Detector")


# ═══════════════════════════════════════════════════════════
# WHITELIST / SAFE LIST
# ═══════════════════════════════════════════════════════════

class SafeList:
    """
    Xavfsiz IP manzillar ro'yxati.
    Bu ro'yxatdagi IP lar hech qachon bloklanmaydi.
    """

    def __init__(self):
        self._ips: Set[str] = set()
        self._subnets: list = []  # (base_ip, prefix_len)
        self.lock = Lock()

        # Standart xavfsiz IP lar
        self._init_defaults()

    def _init_defaults(self):
        """Standart gateway, DNS, loopback qo'shish."""
        # Loopback va broadcast
        default_ips = {
            "127.0.0.1",
            "0.0.0.0",
            "255.255.255.255",
        }

        # Gateway IP larni avtomatik aniqlash
        gateways = self._detect_gateways()
        default_ips.update(gateways)

        # DNS serverlarni aniqlash
        dns_servers = self._detect_dns_servers()
        default_ips.update(dns_servers)

        # Local IP
        local_ip = self._detect_local_ip()
        if local_ip:
            default_ips.add(local_ip)

        self._ips = default_ips

        logger.info(f"🛡️ Safe List: {len(self._ips)} ta IP qo'shildi")
        for ip in sorted(self._ips):
            logger.info(f"   ├── {ip}")

    def _detect_gateways(self) -> set:
        """Default gateway IP larni aniqlash."""
        gateways = set()

        # Ko'p tarqalgan gateway manzillar
        common_gateways = [
            "192.168.0.1", "192.168.1.1", "192.168.2.1",
            "10.0.0.1", "10.0.1.1", "10.1.1.1",
            "172.16.0.1",
        ]

        try:
            import platform
            if platform.system() == "Windows":
                import subprocess
                result = subprocess.run(
                    ["ipconfig"], capture_output=True, text=True, timeout=5
                )
                import re
                # Default Gateway ni topish
                gw_matches = re.findall(
                    r"Default Gateway.*?:\s*(\d+\.\d+\.\d+\.\d+)", result.stdout
                )
                gateways.update(gw_matches)
            else:
                import subprocess
                result = subprocess.run(
                    ["ip", "route", "show", "default"],
                    capture_output=True, text=True, timeout=5
                )
                import re
                gw_matches = re.findall(
                    r"via\s+(\d+\.\d+\.\d+\.\d+)", result.stdout
                )
                gateways.update(gw_matches)
        except Exception:
            pass

        # Agar topilmasa, ko'p tarqalganlarni qo'shamiz
        if not gateways:
            gateways.update(common_gateways[:3])

        return gateways

    def _detect_dns_servers(self) -> set:
        """DNS server IP larni aniqlash."""
        dns_ips = set()

        # Taniqli public DNS
        dns_ips.update({
            "8.8.8.8", "8.8.4.4",        # Google DNS
            "1.1.1.1", "1.0.0.1",        # Cloudflare DNS
            "208.67.222.222",             # OpenDNS
            "9.9.9.9",                    # Quad9
        })

        try:
            import platform
            if platform.system() == "Windows":
                import subprocess
                result = subprocess.run(
                    ["nslookup", "localhost"],
                    capture_output=True, text=True, timeout=5
                )
                import re
                dns_matches = re.findall(
                    r"Address:\s*(\d+\.\d+\.\d+\.\d+)", result.stdout
                )
                dns_ips.update(dns_matches)
        except Exception:
            pass

        return dns_ips

    def _detect_local_ip(self) -> Optional[str]:
        """Lokal IP manzilni aniqlash."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return None

    def add_ip(self, ip: str, reason: str = ""):
        """IP ni safe listga qo'shish."""
        with self.lock:
            self._ips.add(ip)
            logger.info(f"🛡️ Safe List +: {ip} ({reason})")

    def remove_ip(self, ip: str):
        """IP ni safe listdan chiqarish."""
        with self.lock:
            self._ips.discard(ip)

    def is_safe(self, ip: str) -> bool:
        """IP xavfsiz ro'yxatdami tekshirish."""
        with self.lock:
            if ip in self._ips:
                return True

            # Private IP ranglarini tekshirish (ixtiyoriy)
            # 192.168.x.x va 10.x.x.x — default gateway bo'lishi mumkin
            return False

    def add_active_connections(self):
        """Hozirda faol ulanishlarning IP larini safe listga qo'shish."""
        try:
            import subprocess
            import re
            import platform

            if platform.system() == "Windows":
                result = subprocess.run(
                    ["netstat", "-n"], capture_output=True, text=True, timeout=10
                )
            else:
                result = subprocess.run(
                    ["ss", "-tn"], capture_output=True, text=True, timeout=10
                )

            ip_pattern = re.compile(r"(\d+\.\d+\.\d+\.\d+)")
            found_ips = set(ip_pattern.findall(result.stdout))

            # Faqat ochiq portlarga ulanganlarni qo'shamiz
            safe_connections = set()
            for ip in found_ips:
                if not ip.startswith("127.") and not ip.startswith("0."):
                    safe_connections.add(ip)

            with self.lock:
                added = safe_connections - self._ips
                self._ips.update(safe_connections)

            if added:
                logger.info(f"🛡️ Safe List: {len(added)} ta faol ulanish qo'shildi")

        except Exception as e:
            logger.warning(f"⚠️ Active connections: {e}")

    def get_all(self) -> set:
        """Barcha safe IP larni qaytarish."""
        with self.lock:
            return set(self._ips)

    def count(self) -> int:
        with self.lock:
            return len(self._ips)


# ═══════════════════════════════════════════════════════════
# DEBUG FEATURE LOGGER
# ═══════════════════════════════════════════════════════════

class DebugFeatureLogger:
    """
    Modelga kirayotgan feature qiymatlarini debug_log.txt ga yozib boradi.
    False Positive sababini aniqlash uchun foydali.
    """

    def __init__(self, filepath: str = DEBUG_LOG_FILE, max_lines: int = 10000):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self.max_lines = max_lines
        self._line_count = 0
        self.lock = Lock()
        self.enabled = True

        # Header yozish
        self._write_header()

    def _write_header(self):
        """debug_log.txt header yozish."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write("=" * 120 + "\n")
                f.write(f"IoT-Shield Debug Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Model bashorat Feature qiymatlari — False Positive tahlili uchun\n")
                f.write("=" * 120 + "\n\n")

                # Feature nomlari
                header = "Vaqt          | Src_IP          | Bashorat                       | Ishonch | Pkt  | Qaror       | "
                header += " | ".join([f"{f[:8]:>8s}" for f in FEATURE_ORDER])
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
        except Exception as e:
            logger.warning(f"Debug log header xato: {e}")

    def log_prediction(
        self, src_ip: str, label: str, confidence: float,
        n_packets: int, decision: str, features: np.ndarray
    ):
        """Bashoratni debug logga yozish."""
        if not self.enabled:
            return

        with self.lock:
            if self._line_count >= self.max_lines:
                return  # Limit

            try:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
                feat_str = " | ".join([f"{v:>8.2f}" for v in features])

                line = (
                    f"{ts} | {src_ip:<15s} | {label:<30s} | {confidence:>6.1%} | "
                    f"{n_packets:>4d} | {decision:<11s} | {feat_str}\n"
                )

                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(line)

                self._line_count += 1

            except Exception:
                pass

    def log_summary(self, stats: dict):
        """Yakuniy statistikani yozish."""
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"YAKUNIY STATISTIKA — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  Jami paketlar:    {stats.get('total_packets', 0):,}\n")
                f.write(f"  Jami bashoratlar: {stats.get('total_predictions', 0):,}\n")
                f.write(f"  Hujumlar:         {stats.get('total_attacks', 0)}\n")
                f.write(f"  Bloklangan:       {stats.get('total_blocked', 0)}\n")
                f.write(f"  FP saqlangan:     {stats.get('false_positives_prevented', 0)}\n")
                f.write("=" * 80 + "\n")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# MODEL YUKLASH
# ═══════════════════════════════════════════════════════════

def find_latest_model(models_dir: str) -> dict:
    """Eng oxirgi model fayllarini topish."""
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models papkasi topilmadi: {models_dir}")

    meta_files = sorted(
        [f for f in os.listdir(models_dir)
         if f.startswith("metadata_") and f.endswith(".json")],
        reverse=True
    )

    if not meta_files:
        raise FileNotFoundError("Model metadata fayli topilmadi!")

    meta_path = os.path.join(models_dir, meta_files[0])
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    files_info = metadata.get("files", {})
    paths = {}
    for key in ("pkl", "scaler", "label_encoder"):
        fname = files_info.get(key, "")
        fpath = os.path.join(models_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"{key} fayli topilmadi: {fpath}")
        paths[key] = fpath

    paths["metadata"] = metadata
    return paths


def load_model(models_dir: str) -> tuple:
    """Model, scaler, encoder yuklab berish."""
    paths = find_latest_model(models_dir)
    
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

    logger.info("📦 Model yuklanmoqda...")
    model_file = os.path.basename(paths["pkl"])
    
    if "ensemble" in model_file.lower():
        from ensemble import IncrementalEnsemble
        model = IncrementalEnsemble.load(paths["pkl"])
        logger.info(f"   ├── Ensemble model: {model_file}")
        logger.info(f"   ├── Komponentlar: {', '.join(model.models.keys())}")
    else:
        model = joblib.load(paths["pkl"])
        logger.info(f"   ├── Model: {model_file}")

    scaler = joblib.load(paths["scaler"])
    logger.info(f"   ├── Scaler: {os.path.basename(paths['scaler'])}")

    encoder = joblib.load(paths["label_encoder"])
    logger.info(f"   ├── Encoder: {os.path.basename(paths['label_encoder'])}")

    logger.info(f"   ├── Klasslar: {meta['n_classes']}")
    logger.info(f"   ├── Features: {meta['n_features']}")
    logger.info(f"   └── O'qitilgan: {meta.get('total_rows', '?'):,} qator")

    return model, scaler, encoder, meta


# ═══════════════════════════════════════════════════════════
# FLOW WINDOW — Paketlardan feature hisoblash
# ═══════════════════════════════════════════════════════════

class FlowWindow:
    """
    1 soniyalik sliding window.
    Paketlarni yig'ib, CICIOT2023 formatida 37 ta feature qaytaradi.
    """

    __slots__ = [
        "timestamps", "sizes", "header_lengths",
        "src_ip", "dst_ip", "sport", "dport", "proto",
        "tcp_flags", "proto_ports", "lock", "_pkt_count",
    ]

    def __init__(self):
        self.timestamps = deque(maxlen=50000)
        self.sizes = deque(maxlen=50000)
        self.header_lengths = deque(maxlen=50000)
        self.tcp_flags = defaultdict(int)
        self.proto_ports = set()
        self.src_ip = None
        self.dst_ip = None
        self.sport = 0
        self.dport = 0
        self.proto = 0
        self.lock = Lock()
        self._pkt_count = 0

    def add_packet(self, pkt):
        """Paketni oynaga tezkor qo'shish."""
        with self.lock:
            ts = time.time()
            self.timestamps.append(ts)
            self._pkt_count += 1

            # IP qatlam
            if IP in pkt:
                ip_layer = pkt[IP]
                self.src_ip = ip_layer.src
                self.dst_ip = ip_layer.dst
                self.proto = ip_layer.proto
                pkt_size = len(pkt)
                hdr_len = ip_layer.ihl * 4
            elif ARP in pkt:
                self.proto = 0
                self.src_ip = pkt[ARP].psrc
                self.dst_ip = pkt[ARP].pdst
                pkt_size = len(pkt)
                hdr_len = 28
            else:
                pkt_size = len(pkt)
                hdr_len = 14

            self.sizes.append(pkt_size)
            self.header_lengths.append(hdr_len)

            # TCP flaglar
            if TCP in pkt:
                tcp = pkt[TCP]
                self.sport = tcp.sport
                self.dport = tcp.dport
                flags = tcp.flags

                if flags.F: self.tcp_flags["fin"] += 1
                if flags.S: self.tcp_flags["syn"] += 1
                if flags.R: self.tcp_flags["rst"] += 1
                if flags.P: self.tcp_flags["psh"] += 1
                if flags.A: self.tcp_flags["ack"] += 1
                if flags.E: self.tcp_flags["ece"] += 1
                if flags.C: self.tcp_flags["cwr"] += 1
                if flags.U: self.tcp_flags["urg"] += 1

                self.proto_ports.add(tcp.sport)
                self.proto_ports.add(tcp.dport)

            elif UDP in pkt:
                self.sport = pkt[UDP].sport
                self.dport = pkt[UDP].dport
                self.proto_ports.add(pkt[UDP].sport)
                self.proto_ports.add(pkt[UDP].dport)

    def extract_features(self) -> np.ndarray:
        """37 ta CICIOT2023 feature ni hisoblash."""
        with self.lock:
            n = len(self.timestamps)
            if n == 0:
                return np.zeros(len(FEATURE_ORDER), dtype=np.float32)

            ts_arr = np.array(self.timestamps, dtype=np.float64)
            sz_arr = np.array(self.sizes, dtype=np.float32)
            hdr_arr = np.array(self.header_lengths, dtype=np.float32)

            # Vaqt
            duration = max(ts_arr[-1] - ts_arr[0], 0.001)
            rate = n / duration
            iat_mean = float(np.mean(np.diff(ts_arr))) if n > 1 else 0.0

            # Hajm statistikasi
            tot_size = float(np.sum(sz_arr))
            pkt_min = float(np.min(sz_arr))
            pkt_max = float(np.max(sz_arr))
            pkt_avg = float(np.mean(sz_arr))
            pkt_std = float(np.std(sz_arr)) if n > 1 else 0.0
            pkt_var = float(np.var(sz_arr)) if n > 1 else 0.0
            header_len = float(np.mean(hdr_arr))

            # TCP Flaglar
            fin_f = self.tcp_flags.get("fin", 0)
            syn_f = self.tcp_flags.get("syn", 0)
            rst_f = self.tcp_flags.get("rst", 0)
            psh_f = self.tcp_flags.get("psh", 0)
            ack_f = self.tcp_flags.get("ack", 0)
            ece_f = self.tcp_flags.get("ece", 0)
            cwr_f = self.tcp_flags.get("cwr", 0)

            # Port-based protokollar
            ports = self.proto_ports
            proto = self.proto

            feature_map = {
                "ARP": 1 if proto == 0 and self.src_ip else 0,
                "AVG": pkt_avg,
                "DHCP": 1 if ports & {67, 68} else 0,
                "DNS": 1 if ports & {53} else 0,
                "HTTP": 1 if ports & {80, 8080, 8000} else 0,
                "HTTPS": 1 if ports & {443, 8443} else 0,
                "Header_Length": header_len,
                "IAT": iat_mean,
                "ICMP": 1 if proto == 1 else 0,
                "IPv": 1 if proto in (6, 17, 1, 2, 47) else 0,
                "IRC": 1 if ports & {6667, 6668, 6669} else 0,
                "LLC": 0,
                "Max": pkt_max,
                "Min": pkt_min,
                "Number": n,
                "Protocol Type": proto,
                "Rate": rate,
                "SMTP": 1 if ports & {25, 587, 465} else 0,
                "SSH": 1 if ports & {22} else 0,
                "Std": pkt_std,
                "TCP": 1 if proto == 6 else 0,
                "Telnet": 1 if ports & {23} else 0,
                "Tot size": tot_size,
                "Tot sum": tot_size,
                "UDP": 1 if proto == 17 else 0,
                "Variance": pkt_var,
                "ack_count": ack_f,
                "ack_flag_number": 1 if ack_f > 0 else 0,
                "cwr_flag_number": 1 if cwr_f > 0 else 0,
                "ece_flag_number": 1 if ece_f > 0 else 0,
                "fin_count": fin_f,
                "fin_flag_number": 1 if fin_f > 0 else 0,
                "psh_flag_number": 1 if psh_f > 0 else 0,
                "rst_count": rst_f,
                "rst_flag_number": 1 if rst_f > 0 else 0,
                "syn_count": syn_f,
                "syn_flag_number": 1 if syn_f > 0 else 0,
            }

            return np.array(
                [feature_map[f] for f in FEATURE_ORDER], dtype=np.float32
            )

    def get_src_ip(self) -> Optional[str]:
        with self.lock:
            return self.src_ip

    def get_dst_ip(self) -> Optional[str]:
        with self.lock:
            return self.dst_ip

    def get_count(self) -> int:
        with self.lock:
            return len(self.timestamps)

    def reset(self):
        """Oynani tozalash."""
        with self.lock:
            self.timestamps.clear()
            self.sizes.clear()
            self.header_lengths.clear()
            self.tcp_flags.clear()
            self.proto_ports.clear()
            self.src_ip = None
            self.dst_ip = None
            self.sport = 0
            self.dport = 0
            self.proto = 0


# ═══════════════════════════════════════════════════════════
# ASOSIY DETEKTOR
# ═══════════════════════════════════════════════════════════

class DetectionEngine:
    """
    AI Detection Engine v2.0 — False Positive Prevention bilan.

    Qaror mantiqiy:
      confidence >= 95% VA packets >= 10:  → BLOKLASH + Alert
      confidence >= 70% VA packets >= 5:   → ALERT (bloklamasdan)
      confidence < 70% YOKI packets < 5:   → Faqat debug log
      IP safe listda:                      → Hech narsa qilmaslik

    Bot bilan integratsiya:
      engine = DetectionEngine(bot=iot_shield_bot)
      await engine.start_async(interface="eth0")
    """

    def __init__(self, bot=None, interface: str = None,
                 block_threshold: float = BLOCK_THRESHOLD,
                 alert_threshold: float = ALERT_THRESHOLD):
        self.bot = bot
        self.interface = interface
        self.block_threshold = block_threshold
        self.alert_threshold = alert_threshold

        # Model yuklash
        self.model, self.scaler, self.encoder, self.metadata = load_model(MODELS_DIR)
        self.class_names = self.metadata.get("class_names", [])

        # Flow window
        self.flow_window = FlowWindow()

        # Safe List
        self.safe_list = SafeList()

        # Debug Feature Logger
        self.debug_logger = DebugFeatureLogger()

        # Ketma-ket hujum hisoblagichi (IP -> count)
        self._attack_counter: dict = defaultdict(int)
        self._attack_counter_lock = Lock()

        # Statistika
        self.stats = {
            "total_packets": 0,
            "total_predictions": 0,
            "total_attacks": 0,
            "total_blocked": 0,
            "total_alerts": 0,
            "false_positives_prevented": 0,
            "safe_list_skipped": 0,
        }

        self._running = False
        self._loop = None

    async def start_async(self, interface: str = None):
        """Asinxron ishga tushirish."""
        if not SCAPY_OK:
            logger.error("❌ scapy o'rnatilmagan!")
            return

        self.interface = interface or self.interface
        self._running = True
        self._loop = asyncio.get_event_loop()

        conf.verb = 0

        # Faol ulanishlarni safe listga qo'shish
        self.safe_list.add_active_connections()

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║  🧠 AI Detection Engine v2.0 — ishga tushdi      ║")
        logger.info("║  False Positive Prevention + Safe List            ║")
        logger.info("╚══════════════════════════════════════════════════╝")
        logger.info(f"   ├── Interface:       {self.interface or 'barcha'}")
        logger.info(f"   ├── Block Threshold: {self.block_threshold:.0%} (bloklash uchun)")
        logger.info(f"   ├── Alert Threshold: {self.alert_threshold:.0%} (ogohlantirish)")
        logger.info(f"   ├── Min Pkt (block): {MIN_PACKETS_TO_BLOCK}")
        logger.info(f"   ├── Min Pkt (alert): {MIN_PACKETS_TO_ALERT}")
        logger.info(f"   ├── Safe List:       {self.safe_list.count()} ta IP")
        logger.info(f"   ├── Klasslar:        {len(self.class_names)}")
        logger.info(f"   ├── Debug Log:       {DEBUG_LOG_FILE}")
        bot_status = "✅ Ulangan" if self.bot else "❌ Yo'q"
        logger.info(f"   └── Bot:             {bot_status}")
        logger.info("")
        logger.info("🔍 Tarmoq kuzatilmoqda... (Ctrl+C to'xtatish)")
        logger.info("─" * 55)

        # Sniffing thread
        sniff_thread = Thread(target=self._sniff_loop, daemon=True)
        sniff_thread.start()

        # Analysis loop
        await self._analysis_loop_async()

    def _sniff_loop(self):
        """Scapy sniff (blocking, alohida threadda)."""
        try:
            sniff(
                iface=self.interface,
                prn=self._packet_callback,
                store=False,
                stop_filter=lambda _: not self._running,
            )
        except PermissionError:
            logger.error("❌ Root/Admin huquqi kerak!")
        except Exception as e:
            logger.error(f"❌ Sniffing xato: {e}")

    def _packet_callback(self, pkt):
        """Har bir paket — tezkor callback."""
        self.stats["total_packets"] += 1
        self.flow_window.add_packet(pkt)

        if self.bot:
            self.bot.stats["total_packets"] = self.stats["total_packets"]

    async def _analysis_loop_async(self):
        """Har 1 soniyada tahlil."""
        while self._running:
            await asyncio.sleep(WINDOW_SECONDS)

            try:
                n_packets = self.flow_window.get_count()
                if n_packets < 3:
                    continue

                features = self.flow_window.extract_features()
                src_ip = self.flow_window.get_src_ip() or "?"
                dst_ip = self.flow_window.get_dst_ip() or "?"

                self.flow_window.reset()

                await self._predict_and_act(features, src_ip, dst_ip, n_packets)

            except Exception as e:
                logger.warning(f"⚠️ Tahlil xato: {str(e)[:100]}")

    async def _predict_and_act(self, features: np.ndarray,
                                src_ip: str, dst_ip: str, n_packets: int):
        """
        Bashorat + False Positive Prevention mantiqiy.

        Qarorlar darajasi:
          1. SAFE_LIST  — IP xavfsiz ro'yxatda → o'tkazib yuborish
          2. BENIGN     — Model: normal trafik
          3. LOW_CONF   — Hujum lekin ishonch < 70% → faqat debug log
          4. FEW_PKTS   — Paketlar < 5 → faqat debug log
          5. ALERT      — 70-95% ishonch → Telegram alert, bloklamasdan
          6. BLOCK      — >= 95% + >= 10 pkt → BLOKLASH + Alert
          7. REPEAT     — IP 3+ marta ketma-ket → BLOKLASH
        """
        try:
            # ── 1. SAFE LIST tekshirish ──
            if self.safe_list.is_safe(src_ip):
                self.stats["safe_list_skipped"] += 1
                return  # Hech narsa qilmaslik

            # ── 2. Model bashorat ──
            features_extended = add_derived_features_np(features, FEATURE_ORDER)
            X = features_extended.reshape(1, -1)
            X_scaled = self.scaler.transform(X).astype(np.float32)
            prediction = self.model.predict(X_scaled)[0]
            label = self.encoder.inverse_transform([prediction])[0]

            # Ehtimollik
            try:
                proba = self.model.predict_proba(X_scaled)
                confidence = float(np.max(proba))
            except Exception:
                confidence = 0.5  # Noma'lum → past ishonch

            self.stats["total_predictions"] += 1

            if self.bot:
                self.bot.stats["total_predictions"] = self.stats["total_predictions"]

            # ── 3. QAROR QILISH ──

            if label in BENIGN_LABELS:
                # ── NORMAL TRAFIK ──
                decision = "BENIGN"
                self._reset_attack_counter(src_ip)

                # Har 30 soniyada log
                if self.stats["total_predictions"] % 30 == 0:
                    logger.info(
                        f"✅ Normal | Pkt: {n_packets} | "
                        f"Jami: {self.stats['total_predictions']} | "
                        f"Hujumlar: {self.stats['total_attacks']} | "
                        f"FP saqlangan: {self.stats['false_positives_prevented']}"
                    )

                # Debug logga yozish (har 10-chi normal)
                if self.stats["total_predictions"] % 10 == 0:
                    self.debug_logger.log_prediction(
                        src_ip, label, confidence, n_packets, decision, features
                    )

            elif confidence < self.alert_threshold or n_packets < MIN_PACKETS_TO_ALERT:
                # ── PAST ISHONCH / KAM PAKET → False Positive ehtimoli ──
                decision = "FP_PREVENT"
                self.stats["false_positives_prevented"] += 1

                logger.debug(
                    f"🔇 FP prevented | {label} | {confidence:.1%} | "
                    f"Pkt: {n_packets} | IP: {src_ip}"
                )

                # Debug logga YOZISH (tahlil uchun muhim!)
                self.debug_logger.log_prediction(
                    src_ip, label, confidence, n_packets, decision, features
                )

            elif confidence < self.block_threshold or n_packets < MIN_PACKETS_TO_BLOCK:
                # ── O'RTA ISHONCH → ALERT (bloklamasdan) ──
                decision = "ALERT_ONLY"
                self.stats["total_attacks"] += 1
                self.stats["total_alerts"] += 1
                self._increment_attack_counter(src_ip)

                logger.warning(
                    f"⚠️ ALERT | {label:<30s} | "
                    f"IP: {src_ip:<15s} → {dst_ip:<15s} | "
                    f"Ishonch: {confidence:.1%} | Pkt: {n_packets} | "
                    f"Bloklash: YO'Q (threshold ostida)"
                )

                # Debug logga yozish
                self.debug_logger.log_prediction(
                    src_ip, label, confidence, n_packets, decision, features
                )

                # Telegram alert (bloklamasdan)
                if self.bot:
                    self._sync_bot_stats()
                    try:
                        await self.bot.send_attack_alert(
                            attack_type=f"{label} (⚠️ faqat ogohlantirish)",
                            src_ip=src_ip,
                            dst_ip=dst_ip,
                            confidence=confidence,
                            blocked=False,
                            n_packets=n_packets,
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Telegram alert xato: {e}")

                # Agar REPEAT_ATTACK_COUNT ga yetsa — baribir bloklash
                attack_count = self._get_attack_count(src_ip)
                if attack_count >= REPEAT_ATTACK_COUNT:
                    logger.warning(
                        f"🔄 REPEAT ATTACK | {src_ip} — {attack_count} marta ketma-ket! → BLOKLASH"
                    )
                    await self._do_block(src_ip, dst_ip, label, confidence, n_packets, features)

            else:
                # ── YUQORI ISHONCH → BLOKLASH ──
                decision = "BLOCK"
                await self._do_block(src_ip, dst_ip, label, confidence, n_packets, features)

        except Exception as e:
            logger.warning(f"⚠️ Bashorat xato: {str(e)[:120]}")

    async def _do_block(self, src_ip: str, dst_ip: str, label: str,
                         confidence: float, n_packets: int, features: np.ndarray):
        """IP ni bloklash + alert + log."""
        self.stats["total_attacks"] += 1

        # Debug logga yozish
        self.debug_logger.log_prediction(
            src_ip, label, confidence, n_packets, "BLOCK", features
        )

        blocked = False
        if self.bot:
            success, msg = self.bot.firewall.block_ip(src_ip, attack_type=label)
            blocked = success
            if blocked:
                self.stats["total_blocked"] += 1

            self._sync_bot_stats()

            # Telegram xabar
            try:
                await self.bot.send_attack_alert(
                    attack_type=label,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    confidence=confidence,
                    blocked=blocked,
                    n_packets=n_packets,
                )
            except Exception as e:
                logger.warning(f"⚠️ Telegram alert xato: {e}")

        icon = "🔒" if blocked else "⚠️"
        logger.warning(
            f"{icon} HUJUM | {label:<30s} | "
            f"IP: {src_ip:<15s} → {dst_ip:<15s} | "
            f"Ishonch: {confidence:.1%} | Pkt: {n_packets}"
        )

        # Hujum counterini tozalash (bloklangan)
        self._reset_attack_counter(src_ip)

    def _sync_bot_stats(self):
        """Bot statistikasini sync qilish."""
        if self.bot:
            self.bot.stats["total_attacks"] = self.stats["total_attacks"]
            self.bot.stats["total_blocked"] = self.stats["total_blocked"]

    def _increment_attack_counter(self, ip: str):
        with self._attack_counter_lock:
            self._attack_counter[ip] += 1

    def _reset_attack_counter(self, ip: str):
        with self._attack_counter_lock:
            self._attack_counter.pop(ip, None)

    def _get_attack_count(self, ip: str) -> int:
        with self._attack_counter_lock:
            return self._attack_counter.get(ip, 0)

    def stop(self):
        """Detektorni to'xtatish."""
        self._running = False

        # Debug logga yakuniy statistika
        self.debug_logger.log_summary(self.stats)

        logger.info("")
        logger.info("🛑 Detection Engine to'xtatildi")
        logger.info(f"   ├── Jami paketlar:      {self.stats['total_packets']:,}")
        logger.info(f"   ├── Bashoratlar:        {self.stats['total_predictions']:,}")
        logger.info(f"   ├── Hujumlar:           {self.stats['total_attacks']}")
        logger.info(f"   ├── Bloklangan:         {self.stats['total_blocked']}")
        logger.info(f"   ├── Alertlar:           {self.stats['total_alerts']}")
        logger.info(f"   ├── FP saqlangan:       {self.stats['false_positives_prevented']}")
        logger.info(f"   ├── Safe list skip:     {self.stats['safe_list_skipped']}")
        logger.info(f"   └── Debug log:          {DEBUG_LOG_FILE}")


# ═══════════════════════════════════════════════════════════
# STANDALONE REJIM
# ═══════════════════════════════════════════════════════════

def main():
    """Detektorni mustaqil ishga tushirish (bot siz)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(LOG_DIR, "detector.log"), encoding="utf-8"
            ),
        ]
    )

    parser = argparse.ArgumentParser(
        description="🧠 IoT-Shield AI Detection Engine v2.0"
    )
    parser.add_argument("-i", "--interface", default=None)
    parser.add_argument("-b", "--block-threshold", type=float, default=0.95,
                        help="Bloklash uchun minimum ishonch (default: 0.95)")
    parser.add_argument("-a", "--alert-threshold", type=float, default=0.70,
                        help="Alert uchun minimum ishonch (default: 0.70)")
    args = parser.parse_args()

    engine = DetectionEngine(
        bot=None,
        interface=args.interface,
        block_threshold=args.block_threshold,
        alert_threshold=args.alert_threshold,
    )

    try:
        asyncio.run(engine.start_async(args.interface))
    except KeyboardInterrupt:
        engine.stop()


if __name__ == "__main__":
    main()
