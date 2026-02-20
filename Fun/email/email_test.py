import smtplib
import ssl
import time
import os
from datetime import datetime, timezone
from email.message import EmailMessage

# ----------------------------
# CONFIG YOU EDIT
# ----------------------------
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "noah.touchton7788@gmail.com"
SENDER_APP_PASSWORD = "pejcmawxhkpmasgr"

RECIPIENT_EMAIL = "dcl.shoreside.concierge.requests@contact.disneycruiseline.com"
SUBJECT = "Concierge Excursion Requests"

# Text file containing email body (in same directory)
BODY_FILE = "email_body.txt"

TARGET_YEAR = 2026
TARGET_MONTH = 2
TARGET_DAY = 10
TARGET_HOUR = 0
TARGET_MINUTE = 0
TARGET_SECOND = 0

SEND_OFFSET = 0.0
CONNECT_EARLY_SECONDS = 60

# ----------------------------
# HELPERS
# ----------------------------
def local_epoch_seconds(dt_local: datetime) -> float:
    return dt_local.timestamp()

def spin_wait_until(target_epoch: float) -> None:
    while True:
        now = time.time()
        remaining = target_epoch - now
        if remaining <= 0:
            return
        if remaining > 0.2:
            time.sleep(min(remaining - 0.1, 0.5))
        elif remaining > 0.05:
            time.sleep(0.01)
        else:
            pass

def wait_until_with_keepalive(server, target_epoch: float, noop_every=30.0):
    next_noop = time.time() + noop_every
    while True:
        now = time.time()
        remaining = target_epoch - now
        if remaining <= 0:
            return
        if now >= next_noop and remaining > 0.2:
            server.noop()
            next_noop = now + noop_every
        if remaining > 0.2:
            time.sleep(min(remaining - 0.1, 0.5))
        elif remaining > 0.05:
            time.sleep(0.01)
        else:
            pass

def load_email_body(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ----------------------------
# MAIN
# ----------------------------
def main():
    # Load email body from file
    body_text = load_email_body(BODY_FILE)

    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = SUBJECT
    msg.set_content(body_text)

    target_local = datetime(
        TARGET_YEAR, TARGET_MONTH, TARGET_DAY,
        TARGET_HOUR, TARGET_MINUTE, TARGET_SECOND
    )
    target_epoch = local_epoch_seconds(target_local)

    connect_epoch = target_epoch - CONNECT_EARLY_SECONDS

    print("Target local time:", target_local)
    print("Connecting at (epoch):", connect_epoch)
    print("Sending at (epoch):", target_epoch)

    print("Waiting to connect...")
    spin_wait_until(connect_epoch)

    context = ssl.create_default_context()

    print("Connecting to SMTP...")
    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
    server.ehlo()
    server.starttls(context=context)
    server.ehlo()

    print("Logging in...")
    server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)

    print("Connected. Waiting to send...")
    wait_until_with_keepalive(server, target_epoch - SEND_OFFSET, noop_every=20.0)

    send_start = time.time()
    print("Sending NOW:", datetime.now())
    server.send_message(msg)
    send_end = time.time()

    print(f"send_message() took {(send_end - send_start)*1000:.1f} ms")

    try:
        server.quit()
    except Exception:
        server.close()

    print("Done.")

if __name__ == "__main__":
    main()