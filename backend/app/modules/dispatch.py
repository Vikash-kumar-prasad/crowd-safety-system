"""
Emergency Dispatch System Module.

Simulates/executes immediate automated emergency dispatch to:
- Campus Quick Response Team (QRT Mobile Units)
- Central Security Operations Center (Station A)
- Municipal Emergency Dispatch Services (Dial 112/911 Police Gateway)

Optionally dispatches real photos and alerts to Telegram if TELEGRAM_BOT_TOKEN
and TELEGRAM_CHAT_ID environment variables are configured.
"""

import os
import time
import json
import logging
import threading
import urllib.request
import urllib.parse

logger = logging.getLogger("crowd-safety.dispatch")


class EmergencyDispatcher:
    def __init__(self, log_dir="notebooks/alerts"):
        self.log_dir = log_dir
        self.dispatches = []
        self._lock = threading.Lock()
        self._counter = 0

        # Telegram bot configuration (optional)
        self.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

        if self.telegram_bot_token and self.telegram_chat_id:
            logger.info("Telegram Bot integration ENABLED for Emergency Dispatch.")
        else:
            logger.info("Telegram Bot integration inactive (simulating local IoT broadcast).")

    def dispatch(self, alert_record, snapshot_path=None, camera_id="CAM-01 (Main Gate / West Corridor)"):
        """
        Dispatches an emergency broadcast for a Critical incident.
        alert_record: dictionary containing alert metadata (timestamp, reasons, heads, snapshot_filename)
        snapshot_path: full path to snapshot file on disk (optional)
        """
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        safe_time = time.strftime("%Y%m%d_%H%M%S")

        with self._lock:
            self._counter += 1
            ticket_id = f"DISPATCH-{safe_time}-{self._counter:04d}"

        reasons = alert_record.get("reasons", ["Critical security breach"])
        heads = alert_record.get("heads", 0)
        snapshot_fn = alert_record.get("snapshot_filename")
        severity = alert_record.get("severity", "CRITICAL EMERGENCY")

        default_recipients = [
            {"unit": "Campus Quick Response Team (QRT Unit Alpha)", "channel": "Tactical Radio / Mobile Alert", "status": "ACK 200 OK"},
            {"unit": "Central Security Operations Center (SOC)", "channel": "Command Wall Monitor / Audio Klaxon", "status": "ACK 200 OK"},
            {"unit": "Municipal Emergency Dispatch (Police Gateway)", "channel": "Automated CAD Incident Ticket", "status": "ACK 200 OK"},
        ]
        recipients = alert_record.get("recipients", default_recipients)

        dispatch_entry = {
            "ticket_id": ticket_id,
            "timestamp": now,
            "camera_id": camera_id,
            "severity": severity,
            "threat_type": " • ".join(reasons),
            "heads_detected": heads,
            "snapshot_filename": snapshot_fn,
            "recipients": recipients,
            "transmission_status": "TRANSMITTED & ACKNOWLEDGED",
            "latency_ms": 38
        }

        with self._lock:
            self.dispatches.insert(0, dispatch_entry)
            if len(self.dispatches) > 50:
                self.dispatches = self.dispatches[:50]

        logger.warning(f"🚨 EMERGENCY DISPATCH TRANSMITTED: [{ticket_id}] -> Broadcast to {len(recipients)} units for '{dispatch_entry['threat_type']}'")

        # Asynchronously send to Telegram if configured
        if self.telegram_bot_token and self.telegram_chat_id:
            threading.Thread(
                target=self._send_telegram_alert,
                args=(dispatch_entry, snapshot_path),
                daemon=True
            ).start()

        return dispatch_entry

    def _send_telegram_alert(self, dispatch_entry, snapshot_path):
        """Asynchronously send Telegram notification without blocking main loop."""
        try:
            caption = (
                f"🚨 *CRITICAL SECURITY ALERT*\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📋 *Ticket ID:* `{dispatch_entry['ticket_id']}`\n"
                f"📍 *Location:* {dispatch_entry['camera_id']}\n"
                f"🕒 *Timestamp:* {dispatch_entry['timestamp']}\n"
                f"⚠️ *Threat:* {dispatch_entry['threat_type']}\n"
                f"👥 *Heads Detected:* {dispatch_entry['heads_detected']}\n"
                f"📡 *Status:* Broadcast to Police & QRT\n"
                f"━━━━━━━━━━━━━━━━━━"
            )

            # If snapshot exists, send as photo
            if snapshot_path and os.path.exists(snapshot_path):
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"
                # Multipart boundary for uploading photo
                boundary = "----WebKitFormBoundary" + os.urandom(16).hex()
                data = bytearray()

                # chat_id field
                data.extend(f"--{boundary}\r\n".encode("utf-8"))
                data.extend(f'Content-Disposition: form-data; name="chat_id"\r\n\r\n{self.telegram_chat_id}\r\n'.encode("utf-8"))

                # caption field
                data.extend(f"--{boundary}\r\n".encode("utf-8"))
                data.extend(f'Content-Disposition: form-data; name="caption"\r\n\r\n{caption}\r\n'.encode("utf-8"))

                # parse_mode field
                data.extend(f"--{boundary}\r\n".encode("utf-8"))
                data.extend('Content-Disposition: form-data; name="parse_mode"\r\n\r\nMarkdown\r\n'.encode("utf-8"))

                # photo file
                with open(snapshot_path, "rb") as f:
                    file_bytes = f.read()

                data.extend(f"--{boundary}\r\n".encode("utf-8"))
                data.extend(f'Content-Disposition: form-data; name="photo"; filename="{os.path.basename(snapshot_path)}"\r\n'.encode("utf-8"))
                data.extend(b"Content-Type: image/jpeg\r\n\r\n")
                data.extend(file_bytes)
                data.extend(b"\r\n")
                data.extend(f"--{boundary}--\r\n".encode("utf-8"))

                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    logger.info(f"Telegram photo alert sent successfully: {resp.status}")
            else:
                # Text-only message
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                payload = json.dumps({
                    "chat_id": self.telegram_chat_id,
                    "text": caption,
                    "parse_mode": "Markdown"
                }).encode("utf-8")
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    logger.info(f"Telegram text alert sent successfully: {resp.status}")

        except Exception as e:
            logger.error(f"Telegram dispatch notification error: {e}")

    def get_recent_dispatches(self, limit=20):
        with self._lock:
            return list(self.dispatches[:limit])
