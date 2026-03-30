#!/usr/bin/env python3
"""
IoT-Shield Guard — Main Entry Point
=====================================
Telegram Bot + AI Detection Engine ni birga ishga tushiradi.

Ishlatish:
  python guard.py                           # Bot + Detector
  python guard.py --bot-only                # Faqat Bot (test uchun)
  python guard.py --interface eth0          # Aniq interfeys
  python guard.py --chat-id 123456789      # Chat ID belgilash

Raspberry Pi da:
  sudo python3 guard.py --interface eth0
"""

import os
import sys
import logging
import asyncio
import argparse
import platform

# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, "guard.log"),
            encoding="utf-8"
        ),
    ]
)

logger = logging.getLogger("IoT-Shield.Guard")


# ═══════════════════════════════════════════════════════════
# ADMIN TEKSHIRISH
# ═══════════════════════════════════════════════════════════

def check_admin() -> bool:
    """Root/Admin huquqi bormi tekshirish."""
    try:
        if platform.system() == "Windows":
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:
            return os.geteuid() == 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
# ASOSIY FUNKSIYA
# ═══════════════════════════════════════════════════════════

async def run_guard(
    interface: str = None,
    threshold: float = 0.6,
    chat_id: str = None,
    bot_only: bool = False,
):
    """Bot va Detector ni asinxron ishga tushirish."""

    from bot_handler import IoTShieldBot
    from detector import DetectionEngine

    # ── 1. Banner ──
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║        🛡️  IoT-Shield Guard v1.0.0               ║")
    logger.info("║        Real-time Network Intrusion Detection     ║")
    logger.info("║        Telegram Bot + AI Engine                  ║")
    logger.info("╚══════════════════════════════════════════════════╝")
    logger.info("")

    is_admin = check_admin()
    plat = platform.system()
    fw_type = "netsh" if plat == "Windows" else "iptables"

    logger.info(f"   📡 Platforma:   {plat}")
    admin_status = "✅ Ha" if is_admin else "⚠️ Yo'q"
    logger.info(f" 🔑 Admin: {admin_status}")
    logger.info(f"   🔥 Firewall:    {fw_type}")
    logger.info(f"   🌐 Interface:   {interface or 'barcha'}")
    logger.info(f"   📊 Threshold:   {threshold:.0%}")
    logger.info("")

    if not is_admin:
        logger.warning(
            "⚠️  Administrator/root huquqi yo'q!\n"
            "     IP bloklash ishlamaydi. Quyidagicha ishga tushiring:"
        )
        instruktsiya = "Sichqoncha o'ng tugmasi -> Administrator sifatida ishga tushirish" if plat == 'Windows' else "sudo python3 guard.py"
        logger.error(f"      {instruktsiya}")

    # ── 2. Bot yaratish ──
    bot = IoTShieldBot()
    bot.interface = interface
    if chat_id:
        bot.chat_id = chat_id

    # ── 3. Detector yaratish (bot bilan ulash) ──
    engine = None
    if not bot_only:
        try:
            engine = DetectionEngine(
                bot=bot,
                interface=interface,
                block_threshold=0.95,   # 95% dan past — bloklanmaydi
                alert_threshold=0.70,   # 70% dan yuqori — alert
            )
            # Bot va Detector ni o'zaro ulash
            bot._detector = engine
            logger.info("✅ AI Model yuklandi (v2.0 — False Positive Prevention)")
        except FileNotFoundError as e:
            logger.error(f"❌ Model yuklanmadi: {e}")
            logger.info("ℹ️ Bot-only rejimda davom etilmoqda...")
            bot_only = True
        except Exception as e:
            logger.error(f"❌ Detector xato: {e}")
            bot_only = True

    # ── 4. Ishga tushirish ──
    logger.info("")
    logger.info("🚀 IoT-Shield Guard ishga tushmoqda...")
    logger.info("─" * 50)

    tasks = []

    # Bot polling task
    async def start_bot():
        try:
            # Startup xabar
            if bot.chat_id:
                await bot.send_startup_message()
            await bot.start_polling()
        except Exception as e:
            logger.error(f"❌ Bot xato: {e}")

    tasks.append(asyncio.create_task(start_bot()))

    # Detector task
    if engine and not bot_only:
        async def start_engine():
            # Biroz kutish — bot birinchi ishga tushsin
            await asyncio.sleep(2)
            try:
                await engine.start_async(interface)
            except Exception as e:
                logger.error(f"❌ Detector xato: {e}")

        tasks.append(asyncio.create_task(start_engine()))

    # Barcha tasklarni kutish
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        if engine:
            engine.stop()
        await bot.stop()
        logger.info("🛑 IoT-Shield Guard to'xtatildi")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🛡️ IoT-Shield Guard — Telegram Bot + AI NIDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Misollar:
  python guard.py                               # Hammasi birga
  python guard.py --bot-only                     # Faqat bot
  python guard.py -i eth0 -t 0.7                # Raspberry Pi
  python guard.py --chat-id 123456789            # Chat ID belgilash
  sudo python3 guard.py -i wlan0                 # Linux + WiFi
        """
    )

    parser.add_argument(
        "-i", "--interface",
        type=str, default=None,
        help="Tarmoq interfeysi (eth0, wlan0). Default: barcha"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float, default=0.6,
        help="Hujum ishonch darajasi (0.0-1.0). Default: 0.6"
    )
    parser.add_argument(
        "--chat-id",
        type=str, default=None,
        help="Telegram Chat ID (bot /start dan avtomatik aniqlaydi)"
    )
    parser.add_argument(
        "--bot-only",
        action="store_true",
        help="Faqat bot ishga tushsin (detektor siz)"
    )

    args = parser.parse_args()

    try:
        asyncio.run(run_guard(
            interface=args.interface,
            threshold=args.threshold,
            chat_id=args.chat_id,
            bot_only=args.bot_only,
        ))
    except KeyboardInterrupt:
        logger.info("\n🛑 Ctrl+C — IoT-Shield Guard to'xtatildi")
        sys.exit(0)


if __name__ == "__main__":
    main()
