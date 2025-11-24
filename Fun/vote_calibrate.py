import json
import time
from pynput import mouse, keyboard

events = []

start_time = time.time()
last_event_time = start_time

stop_key = 'y'   # stop recording
pause_key = 'p'  # toggle URL marker + pause

paused = False   # True while in "URL typing zone" (not recording clicks/keys)


def rel_timestamp(abs_t: float) -> float:
    """Time since recording started."""
    return abs_t - start_time


def next_dt() -> tuple[float, float]:
    """
    Returns (abs_time, dt) where dt is the time since the previous recorded event.
    Also updates last_event_time.
    """
    global last_event_time
    now = time.time()
    dt = now - last_event_time
    last_event_time = now
    return now, dt


# ------------------ MOUSE HANDLERS ------------------

def on_click(x, y, button, pressed):
    global paused

    # only record button down + when not paused
    if not pressed or paused:
        return

    abs_t, dt = next_dt()

    events.append({
        "type": "mouse_click",
        "button": str(button),
        "position": (x, y),
        "time": rel_timestamp(abs_t),
        "dt": dt,
    })

    print(f"[CLICK] {button} at {(x, y)}  (dt={dt:.3f}s)")


# ------------------ KEYBOARD HANDLERS ------------------

def on_press(key):
    global paused

    try:
        k = key.char  # character keys
    except AttributeError:
        k = str(key)  # special keys like Key.enter

    # ---- global stop, works even when paused ----
    if k == stop_key:
        print("\n[STOP] Stopping recording...\n")
        mouse_listener.stop()
        keyboard_listener.stop()
        return

    # ---- toggle pause + URL marker on 'p' ----
    if k == pause_key:
        if not paused:
            # entering pause: insert URL marker event
            abs_t, dt = next_dt()
            events.append({
                "type": "url_marker",
                "time": rel_timestamp(abs_t),
                "dt": dt,
            })
            paused = True
            print(f"[MARK] URL marker inserted at t={rel_timestamp(abs_t):.3f}s "
                  f"(dt={dt:.3f}s). Recording PAUSED. Press 'p' again to resume.")
        else:
            paused = False
            print("[RESUME] Recording resumed.")
        return

    # ignore all other keys while paused
    if paused:
        return

    # normal key press while recording
    abs_t, dt = next_dt()
    events.append({
        "type": "key_press",
        "key": k,
        "time": rel_timestamp(abs_t),
        "dt": dt,
    })

    print(f"[KEY] {k}  (dt={dt:.3f}s)")


# ------------------ LISTENERS ------------------

with mouse.Listener(on_click=on_click) as mouse_listener, \
     keyboard.Listener(on_press=on_press) as keyboard_listener:

    print("Recording input...")
    print("  'p' -> insert URL marker + pause, 'p' again -> resume")
    print("  'y' -> stop and save\n")

    mouse_listener.join()
    keyboard_listener.join()

# ------------------ SAVE TO JSON ------------------

filename = "input_log.json"
with open(filename, "w") as f:
    json.dump(events, f, indent=4)

print(f"Saved {len(events)} events to {filename}")
