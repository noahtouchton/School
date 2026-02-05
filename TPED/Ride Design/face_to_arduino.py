import time
import cv2
import numpy as np
import requests

# ------------ CONFIG -------------
ARDUINO_BASE = "http://192.168.1.114"   # <-- change
FPS = 60                                # 6–12 is fine
DRAW_MODE = "outline"                   # "outline" or "fill"
PADDING = 0.15                          # expand face box by 15%

# Optional mapping tweaks (use if your matrix is rotated/mirrored vs expectation)
ROTATE = 0                              # 0, 90, 180, 270
FLIP_X = False
FLIP_Y = False

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# ---------------------------------


def apply_orientation(mask_8x12: np.ndarray) -> np.ndarray:
    """mask_8x12 shape (8,12) uint8 0/1"""
    out = (mask_8x12 * 255).astype(np.uint8)

    if ROTATE == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif ROTATE == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif ROTATE == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    out = cv2.resize(out, (12, 8), interpolation=cv2.INTER_NEAREST)

    if FLIP_X:
        out = cv2.flip(out, 1)
    if FLIP_Y:
        out = cv2.flip(out, 0)

    return (out > 0).astype(np.uint8)


def to_bits_rows(mask_8x12: np.ndarray) -> list[str]:
    rows = []
    for y in range(8):
        rows.append("".join("1" if mask_8x12[y, x] else "0" for x in range(12)))
    return rows


def send_rows(rows: list[str]) -> None:
    for y, bits in enumerate(rows):
        requests.get(
            f"{ARDUINO_BASE}/row",
            params={"y": y, "bits": bits},
            timeout=0.35
        )


def largest_face(gray: np.ndarray, face_cascade: cv2.CascadeClassifier):
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])  # x,y,w,h


def draw_bbox_on_12x8(x, y, w, h, img_w, img_h) -> np.ndarray:
    """
    Map a bbox from image coords -> 12x8 coords and draw it on a 12x8 mask.
    Returns mask shape (8,12) with 0/1.
    """
    # Expand bbox
    pad = int(PADDING * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w - 1, x + w + pad)
    y1 = min(img_h - 1, y + h + pad)

    # Normalize to [0,1]
    xn0, yn0 = x0 / img_w, y0 / img_h
    xn1, yn1 = x1 / img_w, y1 / img_h

    # Map to grid coords (12 cols, 8 rows)
    gx0 = int(np.floor(xn0 * 12))
    gy0 = int(np.floor(yn0 * 8))
    gx1 = int(np.ceil(xn1 * 12)) - 1
    gy1 = int(np.ceil(yn1 * 8)) - 1

    # Clamp
    gx0 = max(0, min(11, gx0))
    gx1 = max(0, min(11, gx1))
    gy0 = max(0, min(7, gy0))
    gy1 = max(0, min(7, gy1))

    mask = np.zeros((8, 12), dtype=np.uint8)

    if DRAW_MODE == "fill":
        mask[gy0:gy1+1, gx0:gx1+1] = 1
    else:
        # outline rectangle
        mask[gy0, gx0:gx1+1] = 1
        mask[gy1, gx0:gx1+1] = 1
        mask[gy0:gy1+1, gx0] = 1
        mask[gy0:gy1+1, gx1] = 1

    return mask


def main():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # optional clear
    try:
        requests.get(f"{ARDUINO_BASE}/clear", timeout=0.6)
    except Exception:
        pass

    frame_period = 1.0 / max(1, FPS)
    last_send = 0.0

    print("Running. Keys:")
    print("  q quit | o outline | f fill | r rotate | x flipX | y flipY")

    global DRAW_MODE, ROTATE, FLIP_X, FLIP_Y

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # mirror preview so it feels natural
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]

        face = largest_face(gray, face_cascade)

        # default: blank
        mask = np.zeros((8, 12), dtype=np.uint8)

        if face is not None:
            x, y, w, h = face
            mask = draw_bbox_on_12x8(x, y, w, h, W, H)

            # show ROI on preview for sanity
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{DRAW_MODE} rot={ROTATE} fx={FLIP_X} fy={FLIP_Y}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, "No face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # orientation mapping for your physical matrix
        mask = apply_orientation(mask)

        # debug window: what we’re sending
        dbg = cv2.resize((mask * 255).astype(np.uint8), (12*30, 8*30), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Webcam", frame)
        cv2.imshow("Sending to Arduino (12x8)", dbg)

        now = time.time()
        if now - last_send >= frame_period:
            try:
                send_rows(to_bits_rows(mask))
            except Exception:
                pass
            last_send = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            DRAW_MODE = "outline"
        elif key == ord('f'):
            DRAW_MODE = "fill"
        elif key == ord('r'):
            ROTATE = (ROTATE + 90) % 360
        elif key == ord('x'):
            FLIP_X = not FLIP_X
        elif key == ord('y'):
            FLIP_Y = not FLIP_Y

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()