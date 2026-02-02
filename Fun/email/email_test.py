import smtplib
import ssl
import time
from datetime import datetime, timezone
from email.message import EmailMessage

# ----------------------------
# CONFIG YOU EDIT
# ----------------------------
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587  # STARTTLS port (common for Gmail/Outlook)

SENDER_EMAIL = "noah.touchton7788@gmail.com"
SENDER_APP_PASSWORD = "pejcmawxhkpmasgr"  # Gmail App Password (not your normal password)

RECIPIENT_EMAIL = "ntouchton7@gmail.com"
SUBJECT = "Python Email 726"
BODY = "Hello, this is my submission."

# Target send time (LOCAL time) — example: tonight at 00:00:00
# Change these to the exact date/time you need.
TARGET_YEAR = 2026
TARGET_MONTH = 1
TARGET_DAY = 7
TARGET_HOUR = 19
TARGET_MINUTE = 26
TARGET_SECOND = 0


SEND_OFFSET = 0.5
# How early to connect (seconds). 30–120 is typical.
CONNECT_EARLY_SECONDS = 60

# ----------------------------
# HELPERS
# ----------------------------
def local_epoch_seconds(dt_local: datetime) -> float:
    """
    Convert a *naive* local datetime into epoch seconds by assuming it is local time.
    This avoids timezone complications if you're running on the same local clock you care about.
    """
    # dt_local.timestamp() treats naive datetime as local time in Python
    return dt_local.timestamp()

def spin_wait_until(target_epoch: float) -> None:
    """
    Wait until target time with a two-phase approach:
    - sleep in coarse chunks early (low CPU)
    - spin in the last ~50ms for better precision
    """
    while True:
        now = time.time()
        remaining = target_epoch - now
        if remaining <= 0:
            return
        if remaining > 0.2:
            # Sleep most of the remaining time, but leave a buffer.
            time.sleep(min(remaining - 0.1, 0.5))
        elif remaining > 0.05:
            time.sleep(0.01)
        else:
            # Busy-wait the last ~50ms (more precise, uses CPU briefly)
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

# ----------------------------
# MAIN
# ----------------------------
def main():
    # Build the message now (no delays later)
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = SUBJECT
    msg.set_content(BODY)

    # Compute target time as local datetime, then epoch seconds
    target_local = datetime(
        TARGET_YEAR, TARGET_MONTH, TARGET_DAY,
        TARGET_HOUR, TARGET_MINUTE, TARGET_SECOND
    )
    target_epoch = local_epoch_seconds(target_local)

    # Connect time (open + auth early)
    connect_epoch = target_epoch - CONNECT_EARLY_SECONDS

    print("Target local time:", target_local)
    print("Connecting at (epoch):", connect_epoch)
    print("Sending at (epoch):", target_epoch)

    # Wait until it's time to connect
    print("Waiting to connect...")
    spin_wait_until(connect_epoch)

    context = ssl.create_default_context()

    # Open connection
    print("Connecting to SMTP...")
    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
    server.ehlo()
    server.starttls(context=context)
    server.ehlo()

    # Login
    print("Logging in...")
    server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)

    # Wait until exact send time
    print("Connected. Waiting to send...")
    wait_until_with_keepalive(server, target_epoch - SEND_OFFSET, noop_every=20.0)

    # Send
    send_start = time.time()
    print("Sending NOW:", datetime.now())
    server.send_message(msg)
    send_end = time.time()

    print(f"send_message() took {(send_end - send_start)*1000:.1f} ms")

    # Clean close
    try:
        server.quit()
    except Exception:
        server.close()

    print("Done.")




if __name__ == "__main__":
    main()