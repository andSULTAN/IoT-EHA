"""
IoT-Shield Guard — Telegram Bot Module
========================================
aiogram 3.x asosida Telegram Bot.

Komandalar:
  /start       — Bot haqida ma'lumot
  /status      — Tizim holati
  /scan        — Tarmoqdagi qurilmalarni ARP skan
  /block [IP]  — IP bloklash
  /unblock [IP] — IP blokdan ochish
  /blocked     — Bloklangan IP ro'yxati
  /help        — Yordam

Hujum aniqlanganda — avtomatik xabar + Inline "Blokdan ochish" tugmasi.
"""

import os
import sys
import platform
import asyncio
import subprocess
import logging
import time
from datetime import datetime
from typing import Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    BotCommand
)
from aiogram.filters import Command, CommandStart
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# ═══════════════════════════════════════════════════════════
# KONFIGURATSIYA
# ═══════════════════════════════════════════════════════════

BOT_TOKEN = "8789775060:AAF8UmdufcsJFCfwJkudz8LV9n5Cz4So_ag"
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

logger = logging.getLogger("IoT-Shield.Bot")

# ═══════════════════════════════════════════════════════════
# MAC OUI — Mashhur ishlab chiqaruvchilar (MAC prefix → Nomi)
# ═══════════════════════════════════════════════════════════

MAC_OUI_DB = {
    "00:1A:2B": "Ayecom Technology",
    "00:50:56": "VMware",
    "00:0C:29": "VMware",
    "00:15:5D": "Microsoft (Hyper-V)",
    "08:00:27": "Oracle VirtualBox",
    "B8:27:EB": "Raspberry Pi Foundation",
    "DC:A6:32": "Raspberry Pi Foundation",
    "E4:5F:01": "Raspberry Pi Foundation",
    "D8:3A:DD": "Raspberry Pi (RPi5)",
    "2C:CF:67": "Raspberry Pi (RPi5)",
    "AC:DE:48": "Amazon (Echo/Alexa)",
    "F0:72:EA": "Amazon Technologies",
    "68:54:FD": "Amazon Technologies",
    "50:F5:DA": "Amazon Technologies",
    "18:B4:30": "Nest Labs (Google)",
    "64:16:66": "Google Nest",
    "F4:F5:D8": "Google (Chromecast)",
    "A4:77:33": "Google",
    "30:FD:38": "Google",
    "54:60:09": "Google",
    "3C:5A:B4": "Google (Nest Hub)",
    "7C:2F:80": "Apple",
    "A8:5C:2C": "Apple",
    "F0:18:98": "Apple",
    "D0:D2:B0": "Apple",
    "3C:22:FB": "Apple",
    "F8:1E:DF": "Apple",
    "A4:83:E7": "Apple",
    "CC:50:E3": "Samsung",
    "8C:F5:A3": "Samsung",
    "AC:5F:3E": "Samsung",
    "78:47:1D": "Samsung",
    "00:26:37": "Samsung",
    "EC:FA:BC": "Xiaomi",
    "28:6C:07": "Xiaomi",
    "64:CC:2E": "Xiaomi",
    "78:11:DC": "Xiaomi",
    "50:EC:50": "Xiaomi (Mijia)",
    "7C:49:EB": "Xiaomi",
    "60:AB:67": "TP-Link",
    "C0:06:C3": "TP-Link",
    "14:CC:20": "TP-Link",
    "50:C7:BF": "TP-Link",
    "B0:BE:76": "TP-Link",
    "E8:48:B8": "D-Link",
    "90:94:E4": "D-Link",
    "1C:7E:E5": "D-Link",
    "FC:75:16": "D-Link",
    "B4:75:0E": "Belkin",
    "94:10:3E": "Belkin (WeMo)",
    "C8:3A:35": "Tenda",
    "D8:32:14": "Tenda",
    "00:1E:58": "D-Link",
    "2C:30:33": "NETGEAR",
    "A4:2B:8C": "NETGEAR",
    "6C:B0:CE": "NETGEAR",
    "44:94:FC": "NETGEAR",
    "00:14:6C": "NETGEAR",
    "C8:B5:B7": "Espressif (ESP8266/ESP32)",
    "30:AE:A4": "Espressif (ESP32)",
    "24:0A:C4": "Espressif (ESP32)",
    "AC:67:B2": "Espressif (ESP8266)",
    "A0:20:A6": "Espressif (ESP32-S2/S3)",
    "10:52:1C": "Espressif",
    "A4:CF:12": "Espressif (ESP8266)",
    "84:CC:A8": "Espressif (ESP32)",
    "3C:71:BF": "Espressif (ESP32)",
    "34:AB:95": "Espressif",
    "40:F5:20": "Espressif (ESP32-S3)",
    "48:E7:29": "Espressif (ESP32-C3)",
    "70:04:1D": "Espressif",
    "00:E0:4C": "Realtek",
    "48:5D:60": "Azurewave",
    "00:E0:67": "eQ-3 (HomeMatic)",
    "E0:76:D0": "AMPAK / IoT Module",
    "20:F8:5E": "Delta Electronics",
    "44:67:55": "Orbit (Smart Sprinkler)",
    "D0:73:D5": "LiFi Labs (LIFX Bulb)",
    "D0:03:4B": "Philips Hue",
    "00:17:88": "Philips Hue",
    "EC:B5:FA": "Philips Hue",
    "B0:C5:54": "D-Link / IoT Camera",
    "00:18:DD": "Silicondust (HDHomeRun)",
    "74:DA:38": "Edimax",
    "80:1F:12": "Microchip / IoT",
    "7C:01:0A": "Xiaomi Roborock",
    "44:01:BB": "ShenZhen / IoT",
    "5C:CF:7F": "Espressif (NodeMCU)",
}


# ═══════════════════════════════════════════════════════════
# CROSS-PLATFORM IP BLOKLASH
# ═══════════════════════════════════════════════════════════

class FirewallManager:
    """Windows (netsh) va Linux (iptables) uchun firewall boshqaruv."""

    def __init__(self):
        self.blocked_ips: dict = {}  # ip -> {"time": ts, "attack_type": str}
        self.is_admin = self._check_admin()

        # Whitelist
        self.whitelist = {
            "127.0.0.1", "0.0.0.0", "255.255.255.255",
        }

    def _check_admin(self) -> bool:
        """Administrator/root huquqini tekshirish."""
        try:
            if IS_WINDOWS:
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.geteuid() == 0
        except Exception:
            return False

    def block_ip(self, ip: str, attack_type: str = "Manual") -> tuple:
        """
        IP manzilni bloklash.
        Returns: (success: bool, message: str)
        """
        if ip in self.whitelist:
            return False, f"⚠️ `{ip}` whitelist'da — bloklab bo'lmaydi"

        if ip in self.blocked_ips:
            return False, f"ℹ️ `{ip}` allaqachon bloklangan"

        if not self.is_admin:
            self.blocked_ips[ip] = {
                "time": datetime.now().isoformat(),
                "attack_type": attack_type,
                "firewall": False,
            }
            return True, (
                f"⚠️ `{ip}` ro'yxatga qo'shildi, lekin firewall qoidasi "
                f"qo'shilmadi (Administrator huquqi yo'q).\n"
                f"Dasturni `sudo` bilan qayta ishga tushiring."
            )

        success, msg = self._apply_block(ip)

        if success:
            self.blocked_ips[ip] = {
                "time": datetime.now().isoformat(),
                "attack_type": attack_type,
                "firewall": True,
            }

        return success, msg

    def unblock_ip(self, ip: str) -> tuple:
        """
        IP manzilni blokdan ochish.
        Returns: (success: bool, message: str)
        """
        if ip not in self.blocked_ips:
            return False, f"ℹ️ `{ip}` bloklangan ro'yxatda yo'q"

        if self.blocked_ips[ip].get("firewall", False):
            self._apply_unblock(ip)

        del self.blocked_ips[ip]
        return True, f"✅ `{ip}` blokdan ochildi"

    def _apply_block(self, ip: str) -> tuple:
        """Platformaga mos firewall qoidasini qo'shish."""
        try:
            if IS_WINDOWS:
                rule_name = f"IoT-Shield-Block-{ip.replace('.', '-')}"
                cmd = [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={rule_name}", "dir=in", "action=block",
                    f"remoteip={ip}", "enable=yes"
                ]
            else:
                cmd = ["sudo", "iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                return True, f"🔒 `{ip}` bloklandi ({'netsh' if IS_WINDOWS else 'iptables'})"
            else:
                err = result.stderr.strip()[:150]
                return False, f"❌ Bloklash xato: {err}"

        except subprocess.TimeoutExpired:
            return False, f"❌ Timeout — buyruq bajarilmadi"
        except FileNotFoundError:
            return False, f"❌ Firewall buyrug'i topilmadi"
        except Exception as e:
            return False, f"❌ Xato: {str(e)[:100]}"

    def _apply_unblock(self, ip: str):
        """Firewall qoidasini olib tashlash."""
        try:
            if IS_WINDOWS:
                rule_name = f"IoT-Shield-Block-{ip.replace('.', '-')}"
                cmd = [
                    "netsh", "advfirewall", "firewall", "delete", "rule",
                    f"name={rule_name}"
                ]
            else:
                cmd = ["sudo", "iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"]

            subprocess.run(cmd, capture_output=True, timeout=10)
        except Exception:
            pass

    def get_blocked_list(self) -> dict:
        return dict(self.blocked_ips)


# ═══════════════════════════════════════════════════════════
# ADAPTIVE ARP SCAN — Ko'p subnetli tarmoqlarni aniqlash
# ═══════════════════════════════════════════════════════════

def lookup_mac_vendor(mac: str) -> str:
    """MAC manzilning OUI prefix'i bo'yicha ishlab chiqaruvchini aniqlash."""
    mac_upper = mac.upper().replace("-", ":")
    prefix = mac_upper[:8]  # XX:XX:XX
    return MAC_OUI_DB.get(prefix, "Noma'lum qurilma")


def detect_all_subnets() -> list:
    """
    Tizimdagi barcha tarmoq interfeyslarining subnetlarini aniqlash.
    Returns: ['192.168.0.0/24', '192.168.1.0/24', ...]
    """
    subnets = set()

    try:
        if IS_WINDOWS:
            result = subprocess.run(
                ["ipconfig"], capture_output=True, text=True, timeout=5
            )
            import re
            ip_matches = re.findall(
                r"IPv4.*?:\s*(\d+\.\d+\.\d+)\.\d+", result.stdout
            )
            for base in ip_matches:
                subnets.add(f"{base}.0/24")
        else:
            result = subprocess.run(
                ["ip", "-4", "addr", "show"],
                capture_output=True, text=True, timeout=5
            )
            import re
            ip_matches = re.findall(
                r"inet\s+(\d+\.\d+\.\d+)\.\d+/(\d+)", result.stdout
            )
            for base, prefix in ip_matches:
                if not base.startswith("127"):
                    subnets.add(f"{base}.0/{prefix}")
    except Exception:
        pass

    # Fallback
    if not subnets:
        try:
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            base = ".".join(local_ip.split(".")[:3])
            subnets.add(f"{base}.0/24")
        except Exception:
            subnets.add("192.168.1.0/24")

    return sorted(subnets)


async def arp_scan(interface: str = None, custom_target: str = None) -> list:
    """
    Adaptive ARP skan — bir yoki bir necha subnetni skanerlash.

    custom_target misollar:
      - "192.168.1.0/24"     — bitta subnet
      - "192.168.0.0/16"     — keng diapazon (sekin!)
      - None                 — barcha aniqlangan subnetlar
    """
    try:
        from scapy.all import ARP, Ether, srp, conf

        # Skan targeti aniqlash
        if custom_target:
            targets = [custom_target]
        else:
            targets = detect_all_subnets()

        logger.info(f"🔍 Adaptive ARP Scan: {targets}")

        all_devices = []
        seen_macs = set()  # Dublikatlarni oldini olish

        def _do_scan(target):
            conf.verb = 0
            arp_request = ARP(pdst=target)
            broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
            packet = broadcast / arp_request

            timeout = 3 if "/24" in target else 8  # Keng diapazon uchun uzoqroq
            answered, _ = srp(packet, timeout=timeout, iface=interface, verbose=False)

            devices = []
            for sent, received in answered:
                mac = received.hwsrc.upper()
                if mac in seen_macs:
                    continue
                seen_macs.add(mac)

                ip = received.psrc
                vendor = lookup_mac_vendor(mac)
                devices.append({
                    "ip": ip,
                    "mac": mac,
                    "vendor": vendor,
                    "subnet": target,
                })
            return devices

        # Har bir subnetni skanerlash
        for target in targets:
            try:
                devs = await asyncio.to_thread(_do_scan, target)
                all_devices.extend(devs)
                logger.info(f"   ├── {target}: {len(devs)} qurilma topildi")
            except Exception as e:
                logger.warning(f"   ⚠️ {target}: {e}")

        return sorted(
            all_devices,
            key=lambda d: tuple(int(x) for x in d["ip"].split("."))
        )

    except ImportError:
        logger.warning("scapy kutubxonasi topilmadi")
        return []
    except PermissionError:
        logger.warning("ARP Scan uchun admin huquqi kerak")
        return [{"ip": "—", "mac": "—", "vendor": "⚠️ Admin huquqi kerak", "subnet": ""}]
    except Exception as e:
        logger.error(f"ARP Scan xato: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# TELEGRAM BOT
# ═══════════════════════════════════════════════════════════

class IoTShieldBot:
    """
    IoT-Shield Guard — Telegram Bot.
    Barcha komandalar va callback'lar shu yerda.
    """

    def __init__(self, token: str = BOT_TOKEN, chat_id: str = None):
        self.token = token
        self.chat_id = chat_id  # Auto-detect qilinadi
        self.firewall = FirewallManager()
        self.start_time = datetime.now()
        self.interface = None
        self._detector = None  # DetectionEngine reference (guard.py dan o'rnatiladi)

        # Statistika
        self.stats = {
            "total_attacks": 0,
            "total_blocked": 0,
            "total_packets": 0,
            "total_predictions": 0,
        }

        # Bot va Dispatcher
        self.bot = Bot(
            token=self.token,
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
        )
        self.dp = Dispatcher()
        self.router = Router()
        self.dp.include_router(self.router)

        # Handlerlarni ro'yxatga olish
        self._register_handlers()

    def _register_handlers(self):
        """Barcha komanda va callback handlerlar."""

        @self.router.message(CommandStart())
        async def cmd_start(message: Message):
            self.chat_id = str(message.chat.id)
            platform_name = "🪟 Windows" if IS_WINDOWS else "🐧 Linux"
            admin_status = "✅ Admin" if self.firewall.is_admin else "⚠️ Oddiy"

            text = (
                "🛡 *IoT\\-Shield Guard* — Real\\-vaqt NIDS\n\n"
                "Salom\\! Men tarmoqni real vaqtda kuzataman va "
                "hujumlarni aniqlagan zahoti bloklayman\\.\n\n"
                f"📡 Platforma: {platform_name}\n"
                f"🔑 Huquq: {admin_status}\n"
                f"⏱ Ishga tushdi: `{self.start_time.strftime('%H:%M:%S')}`\n\n"
                "📋 *Komandalar:*\n"
                "/scan — Tarmoqdagi qurilmalar\n"
                "/block `IP` — IP bloklash\n"
                "/unblock `IP` — Blokdan ochish\n"
                "/blocked — Bloklangan ro'yxat\n"
                "/status — Tizim holati\n"
                "/help — Yordam\n"
            )
            await message.answer(text, parse_mode=ParseMode.MARKDOWN_V2)

        @self.router.message(Command("help"))
        async def cmd_help(message: Message):
            text = (
                "📚 *IoT-Shield Guard — Yordam*\n\n"
                "🔍 `/scan` — ARP orqali tarmoqdagi barcha qurilmalarni topadi\n\n"
                "🔒 `/block 192.168.1.50` — Berilgan IP ni bloklaydi\n\n"
                "🔓 `/unblock 192.168.1.50` — IP ni blokdan ochadi\n\n"
                "📋 `/blocked` — Bloklangan IP lar ro'yxati\n\n"
                "📊 `/status` — NIDS holati va statistika\n\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🤖 Bot har bir hujumni aniqlaganda avtomatik xabar yuboradi.\n"
                "Xabar ostidagi 🔓 tugma orqali IP ni blokdan ochish mumkin."
            )
            await message.answer(text)

        @self.router.message(Command("scan"))
        async def cmd_scan(message: Message):
            # /scan yoki /scan 192.168.0.0/16
            parts = message.text.strip().split()
            custom_target = parts[1] if len(parts) > 1 else None

            if custom_target:
                await message.answer(f"🔍 *ARP Scan:* `{custom_target}` ...Kuting.")
            else:
                subnets = detect_all_subnets()
                await message.answer(
                    f"🔍 *Adaptive ARP Scan boshlandi...*\n"
                    f"📡 Subnetlar: `{', '.join(subnets)}`\n"
                    f"Kuting..."
                )

            devices = await arp_scan(
                interface=self.interface, custom_target=custom_target
            )

            if not devices:
                await message.answer("❌ Qurilmalar topilmadi yoki xatolik yuz berdi.")
                return

            # Subnetlar bo'yicha guruhlash
            subnet_groups = {}
            for dev in devices:
                sn = dev.get("subnet", "?")
                if sn not in subnet_groups:
                    subnet_groups[sn] = []
                subnet_groups[sn].append(dev)

            lines = [f"📡 *Tarmoqdagi qurilmalar:* ({len(devices)} ta)\n"]

            for subnet, devs in subnet_groups.items():
                lines.append(f"\n🌐 *{subnet}* ({len(devs)} ta)")
                lines.append("```")
                for i, dev in enumerate(devs, 1):
                    ip = dev["ip"]
                    mac = dev["mac"]
                    vendor = dev["vendor"]
                    lines.append(
                        f"{i:>2}. {ip:<16s} {mac}  {vendor}"
                    )
                lines.append("```")

            text = "\n".join(lines)
            if len(text) > 4000:
                text = text[:3990] + "\n...```"

            await message.answer(text)

        @self.router.message(Command("safelist"))
        async def cmd_safelist(message: Message):
            """Safe List boshqarish: /safelist, /safelist add IP, /safelist remove IP"""
            parts = message.text.strip().split()

            # Agar detector ulangan bo'lsa
            if not hasattr(self, '_detector') or self._detector is None:
                # Safe list mavjud emas — FirewallManager whitelist dan foydalanamiz
                wl = self.firewall.whitelist
                text = (
                    f"🛡 *Safe List:* ({len(wl)} ta)\n\n"
                    + "\n".join([f"  ✅ `{ip}`" for ip in sorted(wl)])
                    + "\n\n_Foydalanish:_\n"
                    + "`/safelist add 192.168.1.50`\n"
                    + "`/safelist remove 192.168.1.50`"
                )
                await message.answer(text)
                return

            safe = self._detector.safe_list

            if len(parts) == 1:
                # Ro'yxatni ko'rsatish
                all_ips = safe.get_all()
                text = f"🛡 *Safe List:* ({len(all_ips)} ta)\n\n"
                for ip in sorted(all_ips):
                    text += f"  ✅ `{ip}`\n"
                text += (
                    "\n_Foydalanish:_\n"
                    "`/safelist add 192.168.1.50`\n"
                    "`/safelist remove 192.168.1.50`"
                )
                await message.answer(text)

            elif len(parts) >= 3 and parts[1].lower() == "add":
                ip = parts[2]
                if self._is_valid_ip(ip):
                    safe.add_ip(ip, reason="Telegram bot")
                    self.firewall.whitelist.add(ip)
                    await message.answer(f"✅ `{ip}` Safe Listga qo'shildi")
                else:
                    await message.answer(f"❌ Noto'g'ri IP: `{ip}`")

            elif len(parts) >= 3 and parts[1].lower() == "remove":
                ip = parts[2]
                safe.remove_ip(ip)
                self.firewall.whitelist.discard(ip)
                await message.answer(f"🗑 `{ip}` Safe Listdan olib tashlandi")

            else:
                await message.answer(
                    "⚠️ Foydalanish:\n"
                    "`/safelist` — ro'yxatni ko'rish\n"
                    "`/safelist add IP` — qo'shish\n"
                    "`/safelist remove IP` — o'chirish"
                )

        @self.router.message(Command("block"))
        async def cmd_block(message: Message):
            parts = message.text.strip().split()
            if len(parts) < 2:
                await message.answer(
                    "⚠️ Foydalanish: `/block 192.168.1.50`",
                )
                return

            ip = parts[1].strip()

            # IP validatsiya
            if not self._is_valid_ip(ip):
                await message.answer(f"❌ Noto'g'ri IP: `{ip}`")
                return

            success, msg = self.firewall.block_ip(ip, attack_type="Manual (Bot)")
            emoji = "🔒" if success else "⚠️"

            # Blokdan ochish tugmasi
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(
                    text="🔓 Blokdan ochish",
                    callback_data=f"unblock:{ip}"
                )]
            ]) if success else None

            await message.answer(f"{emoji} {msg}", reply_markup=keyboard)

        @self.router.message(Command("unblock"))
        async def cmd_unblock(message: Message):
            parts = message.text.strip().split()
            if len(parts) < 2:
                await message.answer("⚠️ Foydalanish: `/unblock 192.168.1.50`")
                return

            ip = parts[1].strip()
            success, msg = self.firewall.unblock_ip(ip)
            await message.answer(msg)

        @self.router.message(Command("blocked"))
        async def cmd_blocked(message: Message):
            blocked = self.firewall.get_blocked_list()

            if not blocked:
                await message.answer("✅ Bloklangan IP manzillar yo'q.")
                return

            lines = [f"🔒 *Bloklangan IP lar:* ({len(blocked)} ta)\n"]
            for ip, info in blocked.items():
                fw = "🔥" if info.get("firewall") else "📝"
                atype = info.get("attack_type", "?")
                t = info.get("time", "?")[:19]
                lines.append(f"{fw} `{ip}` — {atype} ({t})")

            text = "\n".join(lines)
            await message.answer(text)

        @self.router.message(Command("status"))
        async def cmd_status(message: Message):
            uptime = datetime.now() - self.start_time
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)

            platform_name = "Windows" if IS_WINDOWS else "Linux"
            admin = "✅ Ha" if self.firewall.is_admin else "❌ Yo'q"
            blocked_count = len(self.firewall.blocked_ips)

            text = (
                "📊 *IoT-Shield Guard — Status*\n\n"
                f"⏱ Ishlash vaqti: `{hours}s {minutes}d {seconds}s`\n"
                f"📡 Platforma: `{platform_name}`\n"
                f"🔑 Admin huquqi: {admin}\n"
                f"🔒 Bloklangan IP: `{blocked_count}`\n\n"
                f"📦 Jami paketlar: `{self.stats['total_packets']:,}`\n"
                f"🧠 Bashoratlar: `{self.stats['total_predictions']:,}`\n"
                f"⚠️ Hujumlar: `{self.stats['total_attacks']}`\n"
                f"🛡 Bloklashlar: `{self.stats['total_blocked']}`"
            )
            await message.answer(text)

        # ═══ CALLBACK: Inline tugmalar ═══

        @self.router.callback_query(F.data.startswith("unblock:"))
        async def callback_unblock(callback: CallbackQuery):
            ip = callback.data.split(":", 1)[1]
            success, msg = self.firewall.unblock_ip(ip)

            await callback.answer(
                f"{'✅' if success else '⚠️'} {ip} — {'ochildi' if success else 'topilmadi'}",
                show_alert=True
            )

            if success:
                await callback.message.edit_text(
                    callback.message.text + f"\n\n✅ `{ip}` blokdan ochildi.",
                    reply_markup=None,
                )

    # ═══════════════════════════════════════════════════════
    # HUJUM XABARI — Detektordan chaqiriladi
    # ═══════════════════════════════════════════════════════

    async def send_attack_alert(
        self, attack_type: str, src_ip: str, dst_ip: str,
        confidence: float, blocked: bool, n_packets: int = 0
    ):
        """Hujum aniqlanganda Telegram'ga xabar yuborish."""
        if not self.chat_id:
            logger.warning("Chat ID aniqlanmagan — /start buyrug'ini yuboring")
            return

        block_status = "🔒 Bloklandi" if blocked else "⚠️ Bloklanmadi"
        ts = datetime.now().strftime("%H:%M:%S")

        text = (
            f"⚠️ *DIQQAT! Hujum aniqlandi!*\n\n"
            f"🎯 Hujum turi: *{attack_type}*\n"
            f"📡 Manba IP: `{src_ip}`\n"
            f"🎯 Manzil IP: `{dst_ip}`\n"
            f"📊 Ishonch: `{confidence:.1%}`\n"
            f"📦 Paketlar: `{n_packets}`\n"
            f"🕐 Vaqt: `{ts}`\n"
            f"🛡 Holat: {block_status}\n\n"
            f"_Bu xato bo'lsa, quyidagi tugmani bosing:_"
        )

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text="🔓 Blokdan ochish",
                callback_data=f"unblock:{src_ip}"
            )]
        ])

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.error(f"Telegram xabar yuborishda xato: {e}")

    async def send_startup_message(self):
        """Bot ishga tushganda xabar yuborish."""
        if not self.chat_id:
            return

        platform_name = "🪟 Windows" if IS_WINDOWS else "🐧 Linux"
        admin = "✅ Admin" if self.firewall.is_admin else "⚠️ Oddiy"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        text = (
            "🛡 *Aniqlovchi AI tarmoqqa ulandi va himoyani boshladi!*\n\n"
            f"📡 Platforma: {platform_name}\n"
            f"🔑 Huquq: {admin}\n"
            f"⏱ Vaqt: `{ts}`\n\n"
            "Tarmoq real vaqtda kuzatilmoqda... 🔍"
        )

        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            logger.error(f"Startup xabar xato: {e}")

    # ═══════════════════════════════════════════════════════
    # YORDAMCHI
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _is_valid_ip(ip: str) -> bool:
        """Oddiy IP validatsiya."""
        parts = ip.split(".")
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except ValueError:
            return False

    async def start_polling(self):
        """Bot polling ni boshlash."""
        # Bot komandalarini o'rnatish
        commands = [
            BotCommand(command="start", description="Bot haqida"),
            BotCommand(command="scan", description="Tarmoq qurilmalarini skan"),
            BotCommand(command="block", description="IP bloklash"),
            BotCommand(command="unblock", description="IP blokdan ochish"),
            BotCommand(command="blocked", description="Bloklangan ro'yxat"),
            BotCommand(command="safelist", description="Xavfsiz IP ro'yxati"),
            BotCommand(command="status", description="Tizim holati"),
            BotCommand(command="help", description="Yordam"),
        ]

        try:
            await self.bot.set_my_commands(commands)
        except Exception:
            pass

        logger.info("🤖 Telegram Bot ishga tushdi")
        logger.info(f"   Platforma: {'Windows' if IS_WINDOWS else 'Linux'}")
        logger.info(f"   Admin: {self.firewall.is_admin}")

        await self.dp.start_polling(self.bot)

    async def stop(self):
        """Botni to'xtatish."""
        try:
            await self.bot.session.close()
        except Exception:
            pass
