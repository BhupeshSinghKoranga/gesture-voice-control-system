import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# GESTURE DATASET COLLECTOR
# ============================================================
#
# 1 -> INDEX
# 2 -> PINCH
# 3 -> TWO_FINGERS
# 4 -> OPEN_PALM
# 5 -> FIST
# 6 -> THUMBS_UP
# 7 -> THUMBS_DOWN
#
# 500 samples per gesture
# Total = 3500 samples
#
# SPACE = Start / Stop
# R     = Reset current gesture
# Q/ESC = Quit
#
# ============================================================


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "system"
    / "dataset"
    / "gesture_dataset.csv"
)

SAMPLES_PER_GESTURE = 500

# Approximately 25 samples/second.
SAVE_INTERVAL = 0.040

WINDOW_NAME = "Gesture Dataset Collector"

GESTURES = {
    "1": "INDEX",
    "2": "PINCH",
    "3": "TWO_FINGERS",
    "4": "OPEN_PALM",
    "5": "FIST",
    "6": "THUMBS_UP",
    "7": "THUMBS_DOWN",
}


# ============================================================
# COLORS - BGR
# ============================================================

WHITE = (245, 245, 245)
SOFT_WHITE = (205, 210, 215)

GREEN = (80, 230, 130)
RED = (50, 70, 255)

GOLD = (80, 205, 245)

DARK = (8, 12, 18)

GREY = (80, 90, 100)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.60,
    min_tracking_confidence=0.60,
)


# ============================================================
# DATASET DIRECTORY
# ============================================================

DATASET_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CREATE FRESH DATASET
# ============================================================

def create_dataset():

    with open(
        DATASET_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        header = ["label"]

        for i in range(21):
            header.append(f"x_{i}")
            header.append(f"y_{i}")

        writer.writerow(header)


# ============================================================
# OPEN CSV ONCE
# ============================================================

def open_dataset_writer():

    file = open(
        DATASET_FILE,
        "a",
        newline="",
        encoding="utf-8",
        buffering=1
    )

    writer = csv.writer(file)

    return file, writer


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(hand_landmarks):

    points = np.array(
        [
            [
                float(lm.x),
                float(lm.y)
            ]
            for lm in hand_landmarks.landmark
        ],
        dtype=np.float32
    )

    # Must be exactly 21 landmarks.

    if points.shape != (21, 2):

        return None

    # --------------------------------------------------------
    # Wrist as origin
    # --------------------------------------------------------

    points -= points[0].copy()

    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    distances = np.linalg.norm(
        points,
        axis=1
    )

    max_distance = float(
        np.max(distances)
    )

    if max_distance <= 1e-8:

        return None

    points /= max_distance

    # --------------------------------------------------------
    # 21 points x 2 = 42 features
    # --------------------------------------------------------

    features = points.reshape(-1)

    if features.shape[0] != 42:

        return None

    return features.astype(
        np.float32
    )


# ============================================================
# MAXIMIZED WINDOW
# ============================================================

def setup_window():

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    # IMPORTANT:
    # Do NOT use WINDOW_FULLSCREEN.
    # We want Windows taskbar visible.

    try:

        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL
        )

    except Exception:

        pass

    # Give Windows time to create window.

    cv2.waitKey(100)

    # Maximize using Windows API.

    try:

        import ctypes

        user32 = ctypes.windll.user32

        hwnd = user32.FindWindowW(
            None,
            WINDOW_NAME
        )

        if hwnd:

            # SW_MAXIMIZE = 3

            user32.ShowWindow(
                hwnd,
                3
            )

    except Exception:

        pass


# ============================================================
# GET ACTUAL WINDOW CLIENT SIZE
# ============================================================

def get_window_size():

    try:

        import ctypes

        user32 = ctypes.windll.user32

        hwnd = user32.FindWindowW(
            None,
            WINDOW_NAME
        )

        if hwnd:

            class RECT(ctypes.Structure):

                _fields_ = [
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long)
                ]

            rect = RECT()

            user32.GetClientRect(
                hwnd,
                ctypes.byref(rect)
            )

            width = (
                rect.right
                - rect.left
            )

            height = (
                rect.bottom
                - rect.top
            )

            if width > 500 and height > 300:

                return (
                    int(width),
                    int(height)
                )

    except Exception:

        pass

    # Fallback.

    return 1280, 720


# ============================================================
# FILL WINDOW WITHOUT BLACK SIDES
# ============================================================

def fit_frame_to_window(frame):

    target_w, target_h = (
        get_window_size()
    )

    if target_w <= 0 or target_h <= 0:

        return frame

    frame_h, frame_w = (
        frame.shape[:2]
    )

    target_ratio = (
        target_w
        /
        target_h
    )

    current_ratio = (
        frame_w
        /
        frame_h
    )

    # --------------------------------------------------------
    # Camera 4:3 -> crop top/bottom to window ratio.
    # This removes black bars.
    # --------------------------------------------------------

    if current_ratio < target_ratio:

        new_height = int(
            frame_w
            /
            target_ratio
        )

        if new_height < frame_h:

            y1 = (
                frame_h
                - new_height
            ) // 2

            frame = frame[
                y1:y1 + new_height,
                :
            ]

    # --------------------------------------------------------
    # Wider camera -> crop left/right.
    # --------------------------------------------------------

    elif current_ratio > target_ratio:

        new_width = int(
            frame_h
            *
            target_ratio
        )

        if new_width < frame_w:

            x1 = (
                frame_w
                - new_width
            ) // 2

            frame = frame[
                :,
                x1:x1 + new_width
            ]

    # --------------------------------------------------------
    # Resize to actual client area.
    # --------------------------------------------------------

    return cv2.resize(
        frame,
        (
            target_w,
            target_h
        ),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# TEXT
# ============================================================

def draw_text(
    frame,
    text,
    x,
    y,
    scale=0.6,
    thickness=2,
    color=WHITE
):

    cv2.putText(
        frame,
        str(text),
        (
            int(x),
            int(y)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# CENTER TEXT
# ============================================================

def draw_center_text(
    frame,
    text,
    y,
    scale=0.6,
    thickness=2,
    color=WHITE
):

    width = frame.shape[1]

    size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness
    )[0]

    x = (
        width
        - size[0]
    ) // 2

    draw_text(
        frame,
        text,
        x,
        y,
        scale,
        thickness,
        color
    )


# ============================================================
# TRANSPARENT RECTANGLE
# ============================================================

def transparent_rect(
    frame,
    x1,
    y1,
    x2,
    y2,
    color=DARK,
    alpha=0.48
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            int(x1),
            int(y1)
        ),
        (
            int(x2),
            int(y2)
        ),
        color,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ============================================================
# SCAN AREA
# ============================================================

def draw_scan_area(frame):

    height, width = (
        frame.shape[:2]
    )

    # Wide scan area.

    left = int(
        width * 0.06
    )

    right = int(
        width * 0.94
    )

    top = int(
        height * 0.14
    )

    bottom = int(
        height * 0.88
    )

    # Very subtle full scan rectangle.

    cv2.rectangle(
        frame,
        (
            left,
            top
        ),
        (
            right,
            bottom
        ),
        (70, 85, 95),
        1
    )

    # Corner brackets.

    corner = max(
        30,
        int(width * 0.018)
    )

    # --------------------------------------------------------
    # TOP LEFT
    # --------------------------------------------------------

    cv2.line(
        frame,
        (
            left,
            top
        ),
        (
            left + corner,
            top
        ),
        GOLD,
        2
    )

    cv2.line(
        frame,
        (
            left,
            top
        ),
        (
            left,
            top + corner
        ),
        GOLD,
        2
    )

    # --------------------------------------------------------
    # TOP RIGHT
    # --------------------------------------------------------

    cv2.line(
        frame,
        (
            right,
            top
        ),
        (
            right - corner,
            top
        ),
        GOLD,
        2
    )

    cv2.line(
        frame,
        (
            right,
            top
        ),
        (
            right,
            top + corner
        ),
        GOLD,
        2
    )

    # --------------------------------------------------------
    # BOTTOM LEFT
    # --------------------------------------------------------

    cv2.line(
        frame,
        (
            left,
            bottom
        ),
        (
            left + corner,
            bottom
        ),
        GOLD,
        2
    )

    cv2.line(
        frame,
        (
            left,
            bottom
        ),
        (
            left,
            bottom - corner
        ),
        GOLD,
        2
    )

    # --------------------------------------------------------
    # BOTTOM RIGHT
    # --------------------------------------------------------

    cv2.line(
        frame,
        (
            right,
            bottom
        ),
        (
            right - corner,
            bottom
        ),
        GOLD,
        2
    )

    cv2.line(
        frame,
        (
            right,
            bottom
        ),
        (
            right,
            bottom - corner
        ),
        GOLD,
        2
    )


# ============================================================
# UI
# ============================================================

def draw_ui(
    frame,
    gesture,
    count,
    collecting,
    hand_detected
):

    height, width = (
        frame.shape[:2]
    )

    # ========================================================
    # TOP TRANSPARENT HUD
    # ========================================================

    top_h = 82

    transparent_rect(
        frame,
        0,
        0,
        width,
        top_h,
        DARK,
        0.48
    )

    cv2.line(
        frame,
        (0, top_h),
        (width, top_h),
        (90, 105, 115),
        1
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    draw_text(
        frame,
        "GESTURE DATASET",
        28,
        34,
        scale=0.62,
        thickness=2
    )

    draw_text(
        frame,
        "COLLECTOR",
        30,
        59,
        scale=0.39,
        thickness=1,
        color=SOFT_WHITE
    )

    # --------------------------------------------------------
    # GESTURE
    # --------------------------------------------------------

    selected = (
        gesture
        if gesture
        else "NO GESTURE"
    )

    center_x = int(
        width * 0.39
    )

    draw_text(
        frame,
        selected,
        center_x,
        38,
        scale=0.55,
        thickness=2
    )

    draw_text(
        frame,
        f"{count:03d} / 500",
        center_x,
        64,
        scale=0.42,
        thickness=1,
        color=SOFT_WHITE
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    status = (
        "● RECORDING"
        if collecting
        else "● READY"
    )

    status_color = (
        GREEN
        if collecting
        else SOFT_WHITE
    )

    status_size = cv2.getTextSize(
        status,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        2
    )[0]

    draw_text(
        frame,
        status,
        width
        - status_size[0]
        - 30,
        48,
        scale=0.48,
        thickness=2,
        color=status_color
    )

    # ========================================================
    # SCAN AREA
    # ========================================================

    draw_scan_area(
        frame
    )

    # ========================================================
    # HAND STATUS
    # ========================================================

    if not hand_detected:

        draw_center_text(
            frame,
            "SHOW YOUR HAND",
            150,
            scale=0.58,
            thickness=2,
            color=RED
        )

        draw_center_text(
            frame,
            "Place your hand inside the scan area",
            180,
            scale=0.40,
            thickness=1,
            color=SOFT_WHITE
        )

    elif collecting:

        draw_center_text(
            frame,
            "COLLECTING  •  MOVE HAND SLOWLY",
            125,
            scale=0.38,
            thickness=1,
            color=SOFT_WHITE
        )

    # ========================================================
    # BOTTOM TRANSPARENT HUD
    # ========================================================

    bottom_h = 70

    transparent_rect(
        frame,
        0,
        height - bottom_h,
        width,
        height,
        DARK,
        0.52
    )

    # ========================================================
    # BADGES
    # ========================================================

    badges = [
        ("1", "INDEX", "MOVE"),
        ("2", "PINCH", "CLICK"),
        ("3", "TWO", "RIGHT"),
        ("4", "PALM", "SCROLL"),
        ("5", "FIST", "PAUSE"),
        ("6", "UP", "CONFIRM"),
        ("7", "DOWN", "CANCEL"),
    ]

    total_width = int(
        width * 0.82
    )

    start_x = (
        width
        - total_width
    ) // 2

    gap = 8

    badge_width = (
        total_width
        - gap * 6
    ) // 7

    badge_top = (
        height
        - 55
    )

    badge_bottom = (
        height
        - 16
    )

    gesture_match = {
        "INDEX": "INDEX",
        "PINCH": "PINCH",
        "TWO": "TWO_FINGERS",
        "PALM": "OPEN_PALM",
        "FIST": "FIST",
        "UP": "THUMBS_UP",
        "DOWN": "THUMBS_DOWN",
    }

    for i, (
        key,
        name,
        action
    ) in enumerate(badges):

        x1 = (
            start_x
            + i * (
                badge_width
                + gap
            )
        )

        x2 = (
            x1
            + badge_width
        )

        is_active = (
            gesture
            ==
            gesture_match.get(
                name
            )
        )

        # Transparent card.

        transparent_rect(
            frame,
            x1,
            badge_top,
            x2,
            badge_bottom,
            DARK,
            0.52
        )

        border_color = (
            GOLD
            if is_active
            else GREY
        )

        border_thickness = (
            2
            if is_active
            else 1
        )

        cv2.rectangle(
            frame,
            (
                x1,
                badge_top
            ),
            (
                x2,
                badge_bottom
            ),
            border_color,
            border_thickness
        )

        # Key.

        draw_text(
            frame,
            key,
            x1 + 9,
            badge_top + 23,
            scale=0.34,
            thickness=1,
            color=GOLD
        )

        # Gesture.

        draw_text(
            frame,
            name,
            x1 + 29,
            badge_top + 23,
            scale=0.32,
            thickness=1,
            color=WHITE
        )

        # Action.

        draw_text(
            frame,
            action,
            x1 + 9,
            badge_top + 41,
            scale=0.25,
            thickness=1,
            color=SOFT_WHITE
        )

    # ========================================================
    # PROGRESS BAR
    # ========================================================

    progress = (
        count
        /
        SAMPLES_PER_GESTURE
        if gesture
        else 0.0
    )

    progress = max(
        0.0,
        min(
            1.0,
            progress
        )
    )

    bar_width = int(
        width * 0.25
    )

    bar_x = (
        width
        - bar_width
    ) // 2

    bar_y = (
        height
        - bottom_h
        - 7
    )

    cv2.rectangle(
        frame,
        (
            bar_x,
            bar_y
        ),
        (
            bar_x + bar_width,
            bar_y + 4
        ),
        (70, 75, 80),
        -1
    )

    if progress > 0:

        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x
                + int(
                    bar_width
                    * progress
                ),
                bar_y + 4
            ),
            GREEN,
            -1
        )


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    # DirectShow first.

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        camera.release()

        camera = cv2.VideoCapture(
            0
        )

    return camera


# ============================================================
# CAMERA CONFIG
# ============================================================

def configure_camera(camera):

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        30
    )

    # MJPG improves many USB webcam setups.

    try:

        camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            )
        )

    except Exception:

        pass

    # Reduce camera buffering.

    try:

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    except Exception:

        pass


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "              GESTURE DATASET COLLECTOR"
    )
    print("=" * 70)

    print()
    print("GESTURES")
    print()

    for key, gesture in GESTURES.items():

        print(
            f"{key} -> {gesture} "
            f"({SAMPLES_PER_GESTURE} samples)"
        )

    print()

    print(
        f"TOTAL = "
        f"{len(GESTURES) * SAMPLES_PER_GESTURE}"
    )

    print()
    print("CONTROLS")
    print("1-7   Select gesture")
    print("SPACE Start / Stop")
    print("R     Reset counter")
    print("Q     Quit")
    print("ESC   Quit")
    print()

    # ========================================================
    # FRESH DATASET
    # ========================================================

    create_dataset()

    print(
        "Fresh dataset created:"
    )

    print(
        DATASET_FILE
    )

    print()

    # ========================================================
    # CAMERA
    # ========================================================

    camera = open_camera()

    if not camera.isOpened():

        print(
            "ERROR: Camera could not be opened."
        )

        hands.close()

        return

    configure_camera(
        camera
    )

    # ========================================================
    # WINDOW
    # ========================================================

    setup_window()

    # ========================================================
    # STATE
    # ========================================================

    selected_gesture = None

    collecting = False

    sample_count = 0

    last_save_time = 0.0

    csv_file = None

    writer = None

    try:

        # Open CSV once.

        csv_file, writer = (
            open_dataset_writer()
        )

        while True:

            # =================================================
            # CAMERA FRAME
            # =================================================

            success, frame = (
                camera.read()
            )

            if not success:

                time.sleep(
                    0.01
                )

                continue

            # Mirror camera.

            frame = cv2.flip(
                frame,
                1
            )

            # =================================================
            # MEDIAPIPE
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb.flags.writeable = False

            results = hands.process(
                rgb
            )

            rgb.flags.writeable = True

            hand_detected = False

            features = None

            # =================================================
            # HAND
            # =================================================

            if results.multi_hand_landmarks:

                hand_detected = True

                hand = (
                    results.multi_hand_landmarks[0]
                )

                # Draw landmarks.

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                # Extract normalized features.

                features = (
                    extract_features(
                        hand
                    )
                )

            # =================================================
            # COLLECT DATA
            # =================================================

            now = time.perf_counter()

            if (
                collecting
                and
                selected_gesture is not None
                and
                features is not None
                and
                sample_count
                < SAMPLES_PER_GESTURE
                and
                (
                    now
                    - last_save_time
                )
                >= SAVE_INTERVAL
            ):

                writer.writerow(
                    [
                        selected_gesture
                    ]
                    +
                    features.tolist()
                )

                sample_count += 1

                last_save_time = now

                # Save periodically.

                if (
                    sample_count
                    % 20
                    == 0
                ):

                    csv_file.flush()

                # ------------------------------------------------
                # Gesture completed.
                # ------------------------------------------------

                if (
                    sample_count
                    >= SAMPLES_PER_GESTURE
                ):

                    collecting = False

                    csv_file.flush()

                    print()

                    print(
                        f"✓ COMPLETED: "
                        f"{selected_gesture} "
                        f"500/500"
                    )

                    print(
                        "Select next gesture."
                    )

            # =================================================
            # DRAW UI
            # =================================================

            draw_ui(
                frame,
                selected_gesture,
                sample_count,
                collecting,
                hand_detected
            )

            # =================================================
            # MAXIMIZED DISPLAY
            # =================================================

            display = (
                fit_frame_to_window(
                    frame
                )
            )

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            # =================================================
            # KEYBOARD
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # =================================================
            # QUIT
            # =================================================

            if key in (
                ord("q"),
                ord("Q"),
                27
            ):

                break

            # =================================================
            # SELECT GESTURE
            # =================================================

            key_char = (
                chr(key)
                if 0 <= key <= 255
                else ""
            )

            if key_char in GESTURES:

                selected_gesture = (
                    GESTURES[
                        key_char
                    ]
                )

                sample_count = 0

                collecting = False

                print()
                print(
                    "--------------------------------"
                )

                print(
                    f"SELECTED: "
                    f"{selected_gesture}"
                )

                print(
                    "Press SPACE to start."
                )

                print(
                    "--------------------------------"
                )

            # =================================================
            # START / STOP
            # =================================================

            elif key == ord(" "):

                if selected_gesture is None:

                    print(
                        "Select a gesture first."
                    )

                elif (
                    sample_count
                    >= SAMPLES_PER_GESTURE
                ):

                    print(
                        "This gesture is already complete."
                    )

                else:

                    collecting = (
                        not collecting
                    )

                    if collecting:

                        print()
                        print(
                            f"▶ START: "
                            f"{selected_gesture}"
                        )

                        print(
                            "Move your hand slowly."
                        )

                    else:

                        print()
                        print(
                            f"Ⅱ PAUSED: "
                            f"{selected_gesture} "
                            f"{sample_count}/500"
                        )

            # =================================================
            # RESET
            # =================================================

            elif key in (
                ord("r"),
                ord("R")
            ):

                if selected_gesture:

                    sample_count = 0

                    collecting = False

                    print()
                    print(
                        f"Counter reset: "
                        f"{selected_gesture}"
                    )

                    print(
                        "Use R before collecting "
                        "that gesture."
                    )

    finally:

        if csv_file is not None:

            csv_file.flush()
            csv_file.close()

        camera.release()

        cv2.destroyAllWindows()

        hands.close()

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print(
        "           DATASET COLLECTION FINISHED"
    )
    print("=" * 70)

    print()
    print(
        "CSV saved at:"
    )

    print(
        DATASET_FILE
    )

    print()
    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()