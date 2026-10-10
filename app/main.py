import os
import sys
import time
import cv2
import pyautogui

from collections import Counter, deque
from threading import Thread, Lock
import json
import queue
import subprocess
import webbrowser


# ============================================================
# PATHS
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from gesture.hand_detector import HandDetector
from gesture.ml_predictor import MLPredictor
from system.mouse_controller import MouseController

# Optional voice-control dependencies.
# Install: pip install vosk sounddevice pyttsx3
try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    Model = None
    KaldiRecognizer = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


# ============================================================
# CONFIG
# ============================================================

WINDOW_NAME = "Gesture Controlled System"

CAMERA_INDEX = 0

# Capture smaller frame for better webcam throughput.
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# Same size goes directly into MediaPipe.
AI_WIDTH = 320
AI_HEIGHT = 240

# Display size.
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

MIN_CONFIDENCE = 0.45
ACTION_CONFIDENCE = 0.70

# More stable gesture voting.
HISTORY_SIZE = 5
MIN_VOTES = 3

LEFT_MARGIN = 0.05
RIGHT_MARGIN = 0.95
TOP_MARGIN = 0.06
BOTTOM_MARGIN = 0.94

FIST_HOLD_TIME = 1.20
PALM_HOLD_TIME = 0.65

LEFT_CLICK_COOLDOWN = 0.45
RIGHT_CLICK_COOLDOWN = 0.50
SCROLL_INTERVAL = 0.08

# Hand detection can briefly disappear because of motion.
# Keep the last valid landmarks for this short period.
HAND_GRACE_TIME = 0.12

# Landmark smoothing.
LANDMARK_SMOOTHING = 0.42

# ============================================================
# VOICE CONTROL
# ============================================================

VOICE_ENABLED = True

# Put an extracted Vosk English model in one of these folders.
VOICE_MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "models", "vosk-model-small-en-us-0.15"),
    os.path.join(APP_DIR, "models", "vosk-model-small-en-us-0.15"),
]

VOICE_SAMPLE_RATE = 16000
VOICE_DEVICE = 5
VOICE_BLOCK_SIZE = 8000
VOICE_COMMAND_COOLDOWN = 1.20
VOICE_TTS_ENABLED = True


# ============================================================
# COLORS
# ============================================================

WHITE = (245, 245, 245)
LIGHT = (205, 210, 215)
GOLD = (70, 205, 245)
GREEN = (70, 225, 125)
RED = (60, 70, 255)
BLUE = (225, 180, 70)
GREY = (125, 130, 135)
DARK = (15, 18, 22)


# ============================================================
# CAMERA STREAM
# ============================================================

class CameraStream:

    def __init__(self, index=0):

        self.index = index
        self.cap = None

        self.frame = None
        self.frame_id = 0

        self.running = False

        self.lock = Lock()
        self.thread = None

    def start(self):

        # Windows DirectShow first.
        self.cap = cv2.VideoCapture(
            self.index,
            cv2.CAP_DSHOW
        )

        if not self.cap.isOpened():

            try:
                self.cap.release()
            except Exception:
                pass

            self.cap = cv2.VideoCapture(
                self.index
            )

        if not self.cap.isOpened():
            raise RuntimeError(
                "Camera could not be opened."
            )

        # Low-resolution capture for speed.
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            CAMERA_FPS
        )

        # MJPG often gives better USB webcam throughput.
        try:

            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(
                    *"MJPG"
                )
            )

        except Exception:
            pass

        # Low latency buffer.
        try:

            self.cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

        except Exception:
            pass

        self.running = True

        self.thread = Thread(
            target=self._capture,
            daemon=True
        )

        self.thread.start()

        start = time.perf_counter()

        while self.frame is None:

            if (
                time.perf_counter()
                - start
                > 3.0
            ):

                self.stop()

                raise RuntimeError(
                    "Camera opened but no frame received."
                )

            time.sleep(0.003)

        return self

    def _capture(self):

        while self.running:

            if self.cap is None:
                break

            ok, frame = self.cap.read()

            if not ok:
                time.sleep(0.001)
                continue

            # Mirror.
            frame = cv2.flip(
                frame,
                1
            )

            with self.lock:

                # Always replace with newest frame.
                self.frame = frame
                self.frame_id += 1

    def read(self):

        with self.lock:

            if self.frame is None:
                return None, -1

            return (
                self.frame.copy(),
                self.frame_id
            )

    def stop(self):

        self.running = False

        if self.thread is not None:

            if self.thread.is_alive():

                self.thread.join(
                    timeout=1.0
                )

        if self.cap is not None:

            try:
                self.cap.release()
            except Exception:
                pass

            self.cap = None


# ============================================================
# VOICE CONTROLLER
# ============================================================

class VoiceController:

    def __init__(self):
        self.running = False
        self.listening = False
        self.available = False

        self.model = None
        self.recognizer = None
        self.audio_queue = queue.Queue(maxsize=8)

        self.thread = None
        self.tts_thread = None
        self.tts_queue = queue.Queue(maxsize=4)
        self.tts_running = False

        self.lock = Lock()
        self.last_command = ""
        self.status = "VOICE OFF"
        self.last_command_time = 0.0

        self.command_callback = None

    def _tts_loop(self):
        if pyttsx3 is None:
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)

            while self.tts_running:
                try:
                    message = self.tts_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if message:
                    try:
                        engine.say(message)
                        engine.runAndWait()
                    except Exception as error:
                        print(f"TTS error: {error}")

        except Exception as error:
            print(f"TTS initialization failed: {error}")

    def speak(self, message):
        if not VOICE_TTS_ENABLED or pyttsx3 is None:
            return

        if not self.tts_running:
            return

        try:
            self.tts_queue.put_nowait(str(message))
        except queue.Full:
            pass

    def _find_model(self):
        for path in VOICE_MODEL_CANDIDATES:
            if os.path.isdir(path):
                return path
        return None

    def start(self, command_callback):
        self.command_callback = command_callback

        if not VOICE_ENABLED:
            self.status = "VOICE DISABLED"
            return False

        if sd is None or Model is None or KaldiRecognizer is None:
            self.status = "INSTALL VOSK"
            print("Voice disabled: install vosk and sounddevice.")
            return False

        model_path = self._find_model()

        if model_path is None:
            self.status = "MODEL MISSING"
            print("Voice disabled: Vosk model not found.")
            print("Expected model folder:")
            print("models/vosk-model-small-en-us-0.15")
            return False

        
        try:
            self.model = Model(model_path)

            voice_grammar = [
                "open youtube",
                "open you tube",
                "youtube",
                "open google",
                "open chrome",
                "open google chrome",
                "open notepad",
                "open calculator",
                "volume up",
                "increase volume",
                "volume down",
                "decrease volume",
                "mute",
                "scroll up",
                "scroll down",
                "new tab",
                "open new tab",
                "close tab",
                "go back",
                "previous page",
                "refresh page",
                "reload page",
                "take screenshot",
                "screenshot",
                "pause system",
                "pause gesture",
                "resume system",
                "resume gesture",
                "exit program",
                "quit program",
                "stop program",
                "[unk]"
            ]

            self.recognizer = KaldiRecognizer(
                self.model,
                VOICE_SAMPLE_RATE,
                json.dumps(voice_grammar)
            )

            self.running = True
            self.available = True
            self.status = "VOICE READY"

            self.thread = Thread(
                target=self._listen_loop,
                daemon=True
            )
            self.thread.start()

            if VOICE_TTS_ENABLED and pyttsx3 is not None:
                self.tts_running = True
                self.tts_thread = Thread(
                    target=self._tts_loop,
                    daemon=True
                )
                self.tts_thread.start()

            print("Voice control: READY")
            return True

        except Exception as error:
            self.status = "VOICE ERROR"
            print(f"Voice initialization failed: {error}")
            return False


    def _audio_callback(self, indata, frames, time_info, status):
        if not self.running:
            return

        try:
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            # Drop old audio instead of allowing latency to grow.
            pass

    def _listen_loop(self):
        try:
            with sd.RawInputStream(
                   device=VOICE_DEVICE,
                samplerate=VOICE_SAMPLE_RATE,
                blocksize=VOICE_BLOCK_SIZE,
                dtype="int16",
                channels=1,
                callback=self._audio_callback
            ):
                self.listening = True
                self.status = "LISTENING"

                while self.running:
                    try:
                        data = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(
                            self.recognizer.Result()
                        )

                        text = str(
                            result.get("text", "")
                        ).strip().lower()

                        if text:
                            self._handle_recognized_text(text)

        except Exception as error:
            self.status = "MIC ERROR"
            self.listening = False
            print(f"Voice microphone error: {error}")

        finally:
            self.listening = False

    def _handle_recognized_text(self, text):
        now = time.perf_counter()

        if (
            now - self.last_command_time
            < VOICE_COMMAND_COOLDOWN
        ):
            return

        command = self._match_command(text)

        if command is None:
            self.status = f"HEARD: {text[:24]}"
            return

        self.last_command_time = now
        self.last_command = command
        self.status = f"VOICE: {command}"
        self.speak(self._spoken_feedback(command))

        if self.command_callback is not None:
            try:
                self.command_callback(command, text)
            except Exception as error:
                print(f"Voice command error: {error}")

    def _spoken_feedback(self, command):
        feedback = {
            "EXIT": "Exiting system",
            "PAUSE": "System paused",
            "RESUME": "System resumed",
            "SCREENSHOT": "Screenshot saved",
            "VOLUME_UP": "Volume up",
            "VOLUME_DOWN": "Volume down",
            "MUTE": "Mute",
            "SCROLL_UP": "Scroll up",
            "SCROLL_DOWN": "Scroll down",
            "OPEN_CHROME": "Opening Chrome",
            "OPEN_YOUTUBE": "Opening YouTube",
            "OPEN_GOOGLE": "Opening Google",
            "OPEN_NOTEPAD": "Opening Notepad",
            "OPEN_CALCULATOR": "Opening Calculator",
            "NEW_TAB": "New tab",
            "CLOSE_TAB": "Closing tab",
            "GO_BACK": "Going back",
            "REFRESH": "Refreshing page",
        }
        return feedback.get(command, "")

    
    def _match_command(self, text):
        text = text.lower().strip()

        # Exit commands
        if any(p in text for p in (
            "exit program",
            "quit program",
            "close program",
            "stop program",
            "shutdown system"
        )):
            return "EXIT"

        # Pause gesture control
        if any(p in text for p in (
            "pause system",
            "pause gesture",
            "disable gesture"
        )):
            return "PAUSE"

        # Resume gesture control
        if any(p in text for p in (
            "resume system",
            "resume gesture",
            "enable gesture"
        )):
            return "RESUME"

        # Screenshot
        if any(p in text for p in (
            "screenshot",
            "screen shot",
            "take screenshot",
            "take screen shot"
        )):
            return "SCREENSHOT"

        # Volume control
        if any(p in text for p in (
            "volume up",
            "increase volume",
            "louder"
        )):
            return "VOLUME_UP"

        if any(p in text for p in (
            "volume down",
            "decrease volume",
            "quieter"
        )):
            return "VOLUME_DOWN"

        if "mute" in text or "unmute" in text:
            return "MUTE"

        # Scrolling
        if "scroll up" in text:
            return "SCROLL_UP"

        if "scroll down" in text:
            return "SCROLL_DOWN"

        # Browser commands
        if any(p in text for p in (
            "open chrome",
            "open google chrome"
        )):
            return "OPEN_CHROME"

        if any(p in text for p in (
            "open youtube",
            "open you tube",
            "youtube"
        )):
            return "OPEN_YOUTUBE"

        if any(p in text for p in (
            "open google",
            "open google search"
        )):
            return "OPEN_GOOGLE"

        if any(p in text for p in (
            "new tab",
            "open new tab"
        )):
            return "NEW_TAB"

        if "close tab" in text:
            return "CLOSE_TAB"

        if any(p in text for p in (
            "go back",
            "previous page"
        )):
            return "GO_BACK"

        if any(p in text for p in (
            "refresh page",
            "reload page",
            "refresh"
        )):
            return "REFRESH"

        # Applications
        if any(p in text for p in (
            "open notepad",
            "open notes"
        )):
            return "OPEN_NOTEPAD"

        if any(p in text for p in (
            "open calculator",
            "open calc"
        )):
            return "OPEN_CALCULATOR"

        return None

    def stop(self):
        self.running = False
        self.listening = False
        self.tts_running = False

        if self.thread is not None:
            try:
                if self.thread.is_alive():
                    self.thread.join(timeout=1.0)
            except Exception:
                pass

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break

        if self.tts_thread is not None:
            try:
                if self.tts_thread.is_alive():
                    self.tts_thread.join(timeout=1.0)
            except Exception:
                pass

        self.status = "VOICE OFF"

    def get_status(self):
        with self.lock:
            return self.status

# ============================================================
# GESTURE STABILIZER

# ============================================================

class GestureStabilizer:

    def __init__(
        self,
        history_size=5,
        minimum_votes=3
    ):

        self.history = deque(
            maxlen=history_size
        )

        self.minimum_votes = (
            minimum_votes
        )

        self.stable = "NO HAND"

    def update(self, gesture):

        self.history.append(
            gesture
        )

        if not self.history:
            return self.stable

        counts = Counter(
            self.history
        )

        candidate, votes = (
            counts.most_common(1)[0]
        )

        if votes >= self.minimum_votes:

            self.stable = candidate

        return self.stable

    def clear(self):

        self.history.clear()
        self.stable = "NO HAND"


# ============================================================
# LANDMARK SMOOTHER
# ============================================================

class LandmarkSmoother:

    def __init__(
        self,
        alpha=0.42
    ):

        self.alpha = float(alpha)
        self.previous = None

    def update(self, hand):

        if hand is None:

            return self.previous

        try:

            current = [
                (
                    float(lm.x),
                    float(lm.y),
                    float(lm.z)
                )
                for lm in hand.landmark
            ]

            if len(current) != 21:
                return hand

            if self.previous is None:

                self.previous = current

                return hand

            smoothed = []

            for old, new in zip(
                self.previous,
                current
            ):

                x = (
                    old[0]
                    * (1.0 - self.alpha)
                    + new[0]
                    * self.alpha
                )

                y = (
                    old[1]
                    * (1.0 - self.alpha)
                    + new[1]
                    * self.alpha
                )

                z = (
                    old[2]
                    * (1.0 - self.alpha)
                    + new[2]
                    * self.alpha
                )

                smoothed.append(
                    (x, y, z)
                )

            self.previous = smoothed

            # We need a MediaPipe-compatible
            # landmark object for drawing/mouse.
            smoothed_hand = type(
                "SmoothedHand",
                (),
                {}
            )()

            smoothed_hand.landmark = []

            for x, y, z in smoothed:

                lm = type(
                    "Landmark",
                    (),
                    {}
                )()

                lm.x = x
                lm.y = y
                lm.z = z

                smoothed_hand.landmark.append(
                    lm
                )

            return smoothed_hand

        except Exception:

            return hand

    def reset(self):

        self.previous = None


# ============================================================
# DISTANCE
# ============================================================

def point_distance(a, b):

    dx = (
        float(a.x)
        - float(b.x)
    )

    dy = (
        float(a.y)
        - float(b.y)
    )

    return (
        dx * dx
        + dy * dy
    ) ** 0.5


# ============================================================
# INDEX DETECTION
# ============================================================

def is_index_extended(hand):

    if hand is None:
        return False

    try:

        lm = hand.landmark

        if len(lm) != 21:
            return False

        wrist = lm[0]

        mcp = lm[5]
        pip = lm[6]
        dip = lm[7]
        tip = lm[8]

        mcp_d = point_distance(
            mcp,
            wrist
        )

        pip_d = point_distance(
            pip,
            wrist
        )

        dip_d = point_distance(
            dip,
            wrist
        )

        tip_d = point_distance(
            tip,
            wrist
        )

        # Finger should extend progressively.
        increasing = (
            pip_d > mcp_d * 1.08
            and dip_d > pip_d * 0.97
            and tip_d > dip_d * 0.97
        )

        far_enough = (
            tip_d > 0.50
        )

        extended = (
            tip_d
            > mcp_d + 0.18
        )

        return (
            increasing
            and far_enough
            and extended
        )

    except Exception:

        return False


# ============================================================
# REAL FIST
# ============================================================

def is_real_fist(hand):

    if hand is None:
        return False

    try:

        lm = hand.landmark

        if len(lm) != 21:
            return False

        # An extended index means this isn't a fist.
        if is_index_extended(hand):
            return False

        wrist = lm[0]

        tips = [
            lm[8],
            lm[12],
            lm[16],
            lm[20]
        ]

        distances = [
            point_distance(
                tip,
                wrist
            )
            for tip in tips
        ]

        folded = sum(
            d < 0.58
            for d in distances
        )

        return folded >= 3

    except Exception:

        return False


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(frame, hand):

    if hand is None:
        return

    h, w = frame.shape[:2]

    points = []

    for lm in hand.landmark:

        x = int(
            lm.x * w
        )

        y = int(
            lm.y * h
        )

        x = max(
            0,
            min(w - 1, x)
        )

        y = max(
            0,
            min(h - 1, y)
        )

        points.append(
            (x, y)
        )

    if len(points) != 21:
        return

    for a, b in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[a],
            points[b],
            BLUE,
            2,
            cv2.LINE_AA
        )

    for i, p in enumerate(points):

        radius = (
            6
            if i == 8
            else 3
        )

        cv2.circle(
            frame,
            p,
            radius,
            WHITE,
            -1,
            cv2.LINE_AA
        )

    cv2.circle(
        frame,
        points[8],
        13,
        GOLD,
        2,
        cv2.LINE_AA
    )


# ============================================================
# PANEL
# ============================================================

def panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.18
):

    h, w = frame.shape[:2]

    x1 = max(
        0,
        min(w, int(x1))
    )

    x2 = max(
        0,
        min(w, int(x2))
    )

    y1 = max(
        0,
        min(h, int(y1))
    )

    y2 = max(
        0,
        min(h, int(y2))
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return

    roi = frame[
        y1:y2,
        x1:x2
    ]

    overlay = roi.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (
            x2 - x1,
            y2 - y1
        ),
        DARK,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        roi,
        1.0 - alpha,
        0,
        roi
    )


# ============================================================
# TEXT
# ============================================================

def draw_text(
    frame,
    value,
    x,
    y,
    size=0.45,
    thickness=1,
    color=WHITE
):

    cv2.putText(
        frame,
        str(value),
        (
            int(x),
            int(y)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


def center_text(
    frame,
    value,
    y,
    size=0.50,
    thickness=2,
    color=WHITE
):

    h, w = frame.shape[:2]

    (tw, th), _ = cv2.getTextSize(
        str(value),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        thickness
    )

    x = (
        w - tw
    ) // 2

    draw_text(
        frame,
        value,
        x,
        y,
        size,
        thickness,
        color
    )


# ============================================================
# SCAN AREA
# ============================================================

def draw_scan_area(frame):

    h, w = frame.shape[:2]

    left = int(
        w * LEFT_MARGIN
    )

    right = int(
        w * RIGHT_MARGIN
    )

    top = int(
        h * TOP_MARGIN
    )

    bottom = int(
        h * BOTTOM_MARGIN
    )

    cv2.rectangle(
        frame,
        (left, top),
        (right, bottom),
        (90, 100, 105),
        1
    )

    corner = 28

    # Top left
    cv2.line(
        frame,
        (left, top),
        (left + corner, top),
        GOLD,
        3
    )

    cv2.line(
        frame,
        (left, top),
        (left, top + corner),
        GOLD,
        3
    )

    # Top right
    cv2.line(
        frame,
        (right, top),
        (right - corner, top),
        GOLD,
        3
    )

    cv2.line(
        frame,
        (right, top),
        (right, top + corner),
        GOLD,
        3
    )

    # Bottom left
    cv2.line(
        frame,
        (left, bottom),
        (left + corner, bottom),
        GOLD,
        3
    )

    cv2.line(
        frame,
        (left, bottom),
        (left, bottom - corner),
        GOLD,
        3
    )

    # Bottom right
    cv2.line(
        frame,
        (right, bottom),
        (right - corner, bottom),
        GOLD,
        3
    )

    cv2.line(
        frame,
        (right, bottom),
        (right, bottom - corner),
        GOLD,
        3
    )


# ============================================================
# TOP HUD
# ============================================================

def draw_top_hud(
    frame,
    gesture,
    confidence,
    enabled,
    fps,
    ai_ms
):

    h, w = frame.shape[:2]

    panel(
        frame,
        0,
        0,
        w,
        68,
        0.15
    )

    draw_text(
        frame,
        "GESTURE CONTROLLED SYSTEM",
        22,
        29,
        0.46,
        2
    )

    status = (
        "ACTIVE"
        if enabled
        else "PAUSED"
    )

    status_color = (
        GREEN
        if enabled
        else RED
    )

    draw_text(
        frame,
        f"STATUS  {status}",
        22,
        54,
        0.32,
        1,
        status_color
    )

    draw_text(
        frame,
        f"GESTURE  {gesture}",
        int(w * 0.43),
        29,
        0.32,
        1
    )

    draw_text(
        frame,
        f"CONF  {confidence * 100:.0f}%",
        int(w * 0.43),
        54,
        0.29,
        1,
        LIGHT
    )

    draw_text(
        frame,
        f"FPS  {fps:.0f}",
        int(w * 0.78),
        29,
        0.32,
        1
    )

    draw_text(
        frame,
        f"AI  {ai_ms:.0f} ms",
        int(w * 0.78),
        54,
        0.29,
        1,
        LIGHT
    )


# ============================================================
# VOICE HUD
# ============================================================

def draw_voice_hud(frame, voice):
    if voice is None:
        return

    h, w = frame.shape[:2]

    status = voice.get_status()

    if voice.available and voice.listening:
        label = "MIC  LISTENING"
        status_color = GREEN
    elif voice.available:
        label = "MIC  READY"
        status_color = GOLD
    else:
        label = "MIC  OFF"
        status_color = GREY

    panel(
        frame,
        15,
        78,
        235,
        116,
        0.16
    )

    draw_text(
        frame,
        label,
        27,
        98,
        0.31,
        2,
        status_color
    )

    draw_text(
        frame,
        status[:28],
        27,
        112,
        0.22,
        1,
        LIGHT
    )

    draw_text(
        frame,
        "VOICE COMMANDS ACTIVE",
        27,
        132,
        0.20,
        1,
        LIGHT
    )


# ============================================================
# BOTTOM HUD
# ============================================================

def draw_bottom_hud(frame):

    h, w = frame.shape[:2]

    y = h - 65

    panel(
        frame,
        0,
        y,
        w,
        h,
        0.15
    )

    items = [
        ("INDEX", "MOVE"),
        ("PINCH", "CLICK"),
        ("2 FINGERS", "RIGHT"),
        ("PALM", "SCROLL"),
        ("FIST", "PAUSE")
    ]

    gap = 7

    card_width = int(
        (
            w
            - 30
            - gap * 4
        ) / 5
    )

    x = 15

    for title, action in items:

        panel(
            frame,
            x,
            y + 8,
            x + card_width,
            y + 57,
            0.13
        )

        draw_text(
            frame,
            title,
            x + 8,
            y + 30,
            0.28,
            1
        )

        draw_text(
            frame,
            action,
            x + 8,
            y + 49,
            0.21,
            1,
            LIGHT
        )

        x += (
            card_width
            + gap
        )


# ============================================================
# MAIN
# ============================================================

def main():

    camera = None
    detector = None

    try:

        print("=" * 60)
        print("       GESTURE CONTROLLED SYSTEM")
        print("=" * 60)
        print("INDEX       -> Cursor")
        print("PINCH       -> Left Click")
        print("TWO FINGERS -> Right Click")
        print("PALM        -> Scroll / Resume")
        print("FIST        -> Hold to Pause")
        print("F           -> Fullscreen")
        print("VOICE       -> Open apps, volume, browser, pause/resume, etc.")
        print("Q / ESC     -> Exit")
        print("=" * 60)

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        camera = CameraStream(
            CAMERA_INDEX
        ).start()

        # ----------------------------------------------------
        # DETECTOR
        # ----------------------------------------------------

        detector = HandDetector(
            max_hands=1,
            detection_confidence=0.55,
            tracking_confidence=0.55
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model_candidates = [

            os.path.join(
                PROJECT_ROOT,
                "models",
                "gesture_model.pkl"
            ),

            os.path.join(
                PROJECT_ROOT,
                "app",
                "models",
                "gesture_model.pkl"
            ),

            os.path.join(
                PROJECT_ROOT,
                "system",
                "models",
                "gesture_model.pkl"
            )
        ]

        model_path = None

        for path in model_candidates:

            if os.path.isfile(path):

                model_path = path
                break

        if model_path is None:

            raise FileNotFoundError(
                "gesture_model.pkl not found."
            )

        predictor = MLPredictor(
            model_path
        )

        # ----------------------------------------------------
        # MOUSE
        # ----------------------------------------------------

        screen_w, screen_h = (
            pyautogui.size()
        )

        mouse = MouseController(
            screen_w,
            screen_h
        )

        # ----------------------------------------------------
        # VOICE
        # ----------------------------------------------------

        voice = VoiceController()

        # Voice callback is defined after system state exists below.
        voice_command_queue = queue.Queue()

        # ----------------------------------------------------
        # WINDOW
        # ----------------------------------------------------

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        stabilizer = GestureStabilizer(
            HISTORY_SIZE,
            MIN_VOTES
        )

        smoother = LandmarkSmoother(
            LANDMARK_SMOOTHING
        )

        current_gesture = "NO HAND"
        stable_gesture = "NO HAND"
        previous_gesture = "NO HAND"

        confidence = 0.0

        system_enabled = True

        # FIST
        fist_start = None
        fist_triggered = False

        # PALM
        palm_start = None
        palm_triggered = False

        # SCROLL
        previous_palm_y = None
        last_scroll = 0.0

        # CLICK
        last_left_click = 0.0
        last_right_click = 0.0

        # Action toast
        action = "SYSTEM READY"

        action_until = (
            time.perf_counter()
            + 2.0
        )

        # Performance
        fps = 0.0
        fps_counter = 0
        fps_timer = time.perf_counter()

        ai_ms = 0.0

        # ----------------------------------------------------
        # VOICE COMMAND HANDLER
        # ----------------------------------------------------

        voice_action = {
            "text": "VOICE READY",
            "until": time.perf_counter() + 2.0
        }

        def set_voice_action(message):
            voice_action["text"] = message
            voice_action["until"] = (
                time.perf_counter() + 1.6
            )

        def handle_voice_command(command, heard_text):
            voice_command_queue.put(
                (command, heard_text)
            )

        voice.start(handle_voice_command)

        # Detection grace
        last_valid_hand = None
        last_valid_hand_time = 0.0

        # ====================================================
        # LOOP
        # ====================================================

        while True:

            frame, frame_id = (
                camera.read()
            )

            if frame is None:
                continue

            frame_start = time.perf_counter()

            # ------------------------------------------------
            # VOICE COMMANDS
            # ------------------------------------------------

            while True:
                try:
                    command, heard_text = (
                        voice_command_queue.get_nowait()
                    )
                except queue.Empty:
                    break

                now_voice = time.perf_counter()

                if command == "EXIT":
                    set_voice_action("VOICE: EXIT")
                    voice.running = False
                    break

                elif command == "PAUSE":
                    system_enabled = False
                    mouse.reset()
                    previous_palm_y = None
                    fist_start = None
                    fist_triggered = False
                    set_voice_action("VOICE: SYSTEM PAUSED")

                elif command == "RESUME":
                    system_enabled = True
                    mouse.reset()
                    previous_palm_y = None
                    palm_start = None
                    palm_triggered = False
                    set_voice_action("VOICE: SYSTEM RESUMED")

                elif command == "VOLUME_UP":
                    pyautogui.press("volumeup")
                    set_voice_action("VOICE: VOLUME UP")

                elif command == "VOLUME_DOWN":
                    pyautogui.press("volumedown")
                    set_voice_action("VOICE: VOLUME DOWN")

                elif command == "MUTE":
                    pyautogui.press("volumemute")
                    set_voice_action("VOICE: MUTE")

                elif command == "SCROLL_UP":
                    mouse.scroll(5)
                    set_voice_action("VOICE: SCROLL UP")

                elif command == "SCROLL_DOWN":
                    mouse.scroll(-5)
                    set_voice_action("VOICE: SCROLL DOWN")

                elif command == "NEW_TAB":
                    pyautogui.hotkey("ctrl", "t")
                    set_voice_action("VOICE: NEW TAB")

                elif command == "CLOSE_TAB":
                    pyautogui.hotkey("ctrl", "w")
                    set_voice_action("VOICE: CLOSE TAB")

                elif command == "GO_BACK":
                    pyautogui.hotkey("alt", "left")
                    set_voice_action("VOICE: GO BACK")

                elif command == "REFRESH":
                    pyautogui.press("f5")
                    set_voice_action("VOICE: REFRESH")

                elif command == "OPEN_CHROME":
                    try:
                        subprocess.Popen(
                            ["cmd", "/c", "start", "", "chrome"],
                            shell=False
                        )
                    except Exception:
                        webbrowser.open("https://www.google.com")
                    set_voice_action("VOICE: OPEN CHROME")

                elif command == "OPEN_YOUTUBE":
                    webbrowser.open("https://www.youtube.com")
                    set_voice_action("VOICE: OPEN YOUTUBE")

                elif command == "OPEN_GOOGLE":
                    webbrowser.open("https://www.google.com")
                    set_voice_action("VOICE: OPEN GOOGLE")

                elif command == "OPEN_NOTEPAD":
                    try:
                        subprocess.Popen(["notepad.exe"])
                        set_voice_action("VOICE: OPEN NOTEPAD")
                    except Exception:
                        set_voice_action("VOICE: NOTEPAD FAILED")

                elif command == "OPEN_CALCULATOR":
                    try:
                        subprocess.Popen(["calc.exe"])
                        set_voice_action("VOICE: OPEN CALCULATOR")
                    except Exception:
                        set_voice_action("VOICE: CALCULATOR FAILED")

                elif command == "SCREENSHOT":
                    try:
                        stamp = time.strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        screenshot_dir = os.path.join(
                            PROJECT_ROOT,
                            "screenshots"
                        )
                        os.makedirs(
                            screenshot_dir,
                            exist_ok=True
                        )

                        screenshot_path = os.path.join(
                            screenshot_dir,
                            f"screenshot_{stamp}.png"
                        )

                        pyautogui.screenshot(
                            screenshot_path
                        )
                        set_voice_action(
                            "VOICE: SCREENSHOT SAVED"
                        )

                    except Exception:
                        set_voice_action(
                            "VOICE: SCREENSHOT FAILED"
                        )

            # If EXIT was received, leave the loop.
            if not voice.running and voice.last_command == "EXIT":
                break

            # ------------------------------------------------
            # AI
            # ------------------------------------------------

            # Camera is already 320x240.
            small = frame

            ai_start = time.perf_counter()

            results = detector.find_hands(
                small
            )

            raw_hand = None

            if (
                results is not None
                and getattr(
                    results,
                    "multi_hand_landmarks",
                    None
                )
            ):

                raw_hand = (
                    results
                    .multi_hand_landmarks[0]
                )

                last_valid_hand = raw_hand
                last_valid_hand_time = (
                    time.perf_counter()
                )

            # ------------------------------------------------
            # DETECTION GRACE
            # ------------------------------------------------

            if raw_hand is not None:

                hand = smoother.update(
                    raw_hand
                )

            else:

                elapsed_since_hand = (
                    time.perf_counter()
                    - last_valid_hand_time
                )

                if (
                    last_valid_hand is not None
                    and elapsed_since_hand
                    <= HAND_GRACE_TIME
                ):

                    # Keep smoothed hand briefly.
                    hand = smoother.update(
                        last_valid_hand
                    )

                else:

                    hand = None
                    smoother.reset()

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            if raw_hand is not None:

                predicted, pred_conf = (
                    predictor.predict(
                        raw_hand
                    )
                )

                current_gesture = str(
                    predicted
                )

                confidence = float(
                    pred_conf
                )

            elif hand is None:

                current_gesture = "NO HAND"
                confidence = 0.0

            # Don't immediately throw away
            # last recognized gesture during
            # a tiny detection dropout.

            if hand is not None:

                if confidence >= MIN_CONFIDENCE:

                    stable_gesture = (
                        stabilizer.update(
                            current_gesture
                        )
                    )

                elif raw_hand is not None:

                    stable_gesture = (
                        stabilizer.update(
                            "UNKNOWN"
                        )
                    )

            else:

                stable_gesture = (
                    stabilizer.update(
                        "NO HAND"
                    )
                )

            ai_ms = (
                time.perf_counter()
                - ai_start
            ) * 1000.0

            # ------------------------------------------------
            # INDEX SAFETY
            # ------------------------------------------------

            index_override = (
                raw_hand is not None
                and is_index_extended(
                    raw_hand
                )
            )

            # Only override when actual
            # geometric index is clearly extended.

            if index_override:

                current_gesture = "INDEX"

                stable_gesture = "INDEX"

                confidence = max(
                    confidence,
                    0.80
                )

                # Prevent old action states.
                fist_start = None
                fist_triggered = False

                palm_start = None
                palm_triggered = False

            # ------------------------------------------------
            # FIST PAUSE
            # ------------------------------------------------

            if not index_override:

                real_fist = (
                    raw_hand is not None
                    and current_gesture == "FIST"
                    and stable_gesture == "FIST"
                    and confidence >= ACTION_CONFIDENCE
                    and is_real_fist(raw_hand)
                )

                if not real_fist:

                    fist_start = None
                    fist_triggered = False

                else:

                    if fist_start is None:

                        fist_start = (
                            time.perf_counter()
                        )

                    if (
                        time.perf_counter()
                        - fist_start
                        >= FIST_HOLD_TIME
                        and not fist_triggered
                    ):

                        system_enabled = False

                        fist_triggered = True

                        mouse.reset()

                        previous_palm_y = None

                        action = "SYSTEM PAUSED"

                        action_until = (
                            time.perf_counter()
                            + 1.5
                        )

            # ------------------------------------------------
            # INDEX CURSOR
            # ------------------------------------------------

            if (
                system_enabled
                and hand is not None
                and current_gesture == "INDEX"
                and confidence >= 0.45
            ):

                index = hand.landmark[8]

                mouse.move_cursor(
                    index.x * AI_WIDTH,
                    index.y * AI_HEIGHT,
                    AI_WIDTH,
                    AI_HEIGHT,
                    LEFT_MARGIN,
                    RIGHT_MARGIN,
                    TOP_MARGIN,
                    BOTTOM_MARGIN
                )

            # ------------------------------------------------
            # LEFT CLICK
            # ------------------------------------------------

            if (
                not index_override
                and system_enabled
                and stable_gesture == "PINCH"
                and confidence >= ACTION_CONFIDENCE
                and previous_gesture != "PINCH"
            ):

                now = time.perf_counter()

                if (
                    now
                    - last_left_click
                    >= LEFT_CLICK_COOLDOWN
                ):

                    if mouse.left_click():

                        last_left_click = now

                        action = "LEFT CLICK"

                        action_until = (
                            now + 0.8
                        )

            # ------------------------------------------------
            # RIGHT CLICK
            # ------------------------------------------------

            if (
                not index_override
                and system_enabled
                and stable_gesture == "TWO_FINGERS"
                and confidence >= ACTION_CONFIDENCE
                and previous_gesture != "TWO_FINGERS"
            ):

                now = time.perf_counter()

                if (
                    now
                    - last_right_click
                    >= RIGHT_CLICK_COOLDOWN
                ):

                    if mouse.right_click():

                        last_right_click = now

                        action = (
                            "RIGHT CLICK"
                        )

                        action_until = (
                            now + 0.8
                        )

            # ------------------------------------------------
            # PALM RESUME
            # ------------------------------------------------

            real_palm = (
                not system_enabled
                and raw_hand is not None
                and current_gesture == "OPEN_PALM"
                and stable_gesture == "OPEN_PALM"
                and confidence >= 0.60
            )

            if real_palm:

                if palm_start is None:

                    palm_start = (
                        time.perf_counter()
                    )

                if (
                    time.perf_counter()
                    - palm_start
                    >= PALM_HOLD_TIME
                    and not palm_triggered
                ):

                    system_enabled = True

                    palm_triggered = True

                    mouse.reset()

                    fist_start = None
                    fist_triggered = False

                    previous_palm_y = None

                    action = (
                        "SYSTEM RESUMED"
                    )

                    action_until = (
                        time.perf_counter()
                        + 1.2
                    )

            else:

                palm_start = None
                palm_triggered = False

            # ------------------------------------------------
            # SCROLL
            # ------------------------------------------------

            if (
                not index_override
                and system_enabled
                and raw_hand is not None
                and current_gesture == "OPEN_PALM"
                and confidence >= 0.55
            ):

                palm_y = (
                    hand.landmark[0].y
                    + hand.landmark[5].y
                    + hand.landmark[9].y
                    + hand.landmark[13].y
                    + hand.landmark[17].y
                ) / 5.0

                if previous_palm_y is not None:

                    delta = (
                        palm_y
                        - previous_palm_y
                    )

                    now = time.perf_counter()

                    if (
                        abs(delta) >= 0.007
                        and
                        now
                        - last_scroll
                        >= SCROLL_INTERVAL
                    ):

                        amount = int(
                            -delta * 120
                        )

                        if amount == 0:

                            amount = (
                                -1
                                if delta > 0
                                else 1
                            )

                        amount = max(
                            -5,
                            min(5, amount)
                        )

                        if mouse.scroll(
                            amount
                        ):

                            last_scroll = now

                previous_palm_y = palm_y

            else:

                previous_palm_y = None

            # =================================================
            # DISPLAY
            # =================================================

            display = cv2.resize(
                frame,
                (
                    DISPLAY_WIDTH,
                    DISPLAY_HEIGHT
                ),
                interpolation=cv2.INTER_LINEAR
            )

            # Scan area.
            draw_scan_area(
                display
            )

            # Smooth hand.
            if hand is not None:

                draw_hand(
                    display,
                    hand
                )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            fps_counter += 1

            now = time.perf_counter()

            elapsed = (
                now - fps_timer
            )

            if elapsed >= 1.0:

                fps = (
                    fps_counter
                    / elapsed
                )

                fps_counter = 0
                fps_timer = now

            # ------------------------------------------------
            # VOICE HUD
            # ------------------------------------------------

            draw_voice_hud(
                display,
                voice
            )

            # ------------------------------------------------
            # TOP HUD
            # ------------------------------------------------

            draw_top_hud(
                display,
                current_gesture,
                confidence,
                system_enabled,
                fps,
                ai_ms
            )

            # ------------------------------------------------
            # CENTER MESSAGE
            # ------------------------------------------------

            if hand is None:

                center_text(
                    display,
                    "SHOW YOUR HAND",
                    int(
                        DISPLAY_HEIGHT
                        * 0.44
                    ),
                    0.55,
                    2,
                    RED
                )

                center_text(
                    display,
                    "Place your hand inside the scan area",
                    int(
                        DISPLAY_HEIGHT
                        * 0.50
                    ),
                    0.32,
                    1,
                    LIGHT
                )

            elif not system_enabled:

                center_text(
                    display,
                    "SYSTEM PAUSED",
                    int(
                        DISPLAY_HEIGHT
                        * 0.44
                    ),
                    0.55,
                    2,
                    RED
                )

                center_text(
                    display,
                    "HOLD OPEN PALM TO RESUME",
                    int(
                        DISPLAY_HEIGHT
                        * 0.50
                    ),
                    0.32,
                    1,
                    LIGHT
                )

            else:

                messages = {

                    "INDEX":
                        "MOVE CURSOR",

                    "PINCH":
                        "LEFT CLICK",

                    "TWO_FINGERS":
                        "RIGHT CLICK",

                    "OPEN_PALM":
                        "SCROLL",

                    "FIST":
                        "HOLD TO PAUSE"
                }

                msg = messages.get(
                    current_gesture
                )

                if msg:

                    center_text(
                        display,
                        msg,
                        int(
                            DISPLAY_HEIGHT
                            * 0.44
                        ),
                        0.44,
                        2,
                        WHITE
                    )

            # ------------------------------------------------
            # ACTION TOAST
            # ------------------------------------------------

            if (
                time.perf_counter()
                < action_until
            ):

                panel(
                    display,
                    int(
                        DISPLAY_WIDTH
                        * 0.35
                    ),
                    int(
                        DISPLAY_HEIGHT
                        * 0.66
                    ),
                    int(
                        DISPLAY_WIDTH
                        * 0.65
                    ),
                    int(
                        DISPLAY_HEIGHT
                        * 0.73
                    ),
                    0.18
                )

                center_text(
                    display,
                    action,
                    int(
                        DISPLAY_HEIGHT
                        * 0.71
                    ),
                    0.34,
                    1,
                    WHITE
                )

            # ------------------------------------------------
            # VOICE ACTION TOAST
            # ------------------------------------------------

            if (
                time.perf_counter()
                < voice_action["until"]
            ):
                panel(
                    display,
                    int(DISPLAY_WIDTH * 0.28),
                    int(DISPLAY_HEIGHT * 0.78),
                    int(DISPLAY_WIDTH * 0.72),
                    int(DISPLAY_HEIGHT * 0.85),
                    0.18
                )

                center_text(
                    display,
                    voice_action["text"],
                    int(DISPLAY_HEIGHT * 0.83),
                    0.30,
                    1,
                    WHITE
                )

            # ------------------------------------------------
            # BOTTOM HUD
            # ------------------------------------------------

            draw_bottom_hud(
                display
            )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if key in (
                ord("q"),
                ord("Q"),
                27
            ):

                break

            # ------------------------------------------------
            # FULLSCREEN
            # ------------------------------------------------

            if key in (
                ord("f"),
                ord("F")
            ):

                try:

                    current = (
                        cv2.getWindowProperty(
                            WINDOW_NAME,
                            cv2.WND_PROP_FULLSCREEN
                        )
                    )

                    if (
                        current
                        == cv2.WINDOW_FULLSCREEN
                    ):

                        cv2.setWindowProperty(
                            WINDOW_NAME,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_NORMAL
                        )

                        cv2.resizeWindow(
                            WINDOW_NAME,
                            960,
                            720
                        )

                    else:

                        cv2.setWindowProperty(
                            WINDOW_NAME,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN
                        )

                except Exception:
                    pass

            # IMPORTANT:
            # Previous gesture is updated only
            # after all action checks.
            previous_gesture = (
                stable_gesture
            )

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

    except Exception as error:

        print()
        print("=" * 60)
        print("APPLICATION ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)

    finally:

        try:
            if "voice" in locals() and voice is not None:
                voice.stop()
        except Exception:
            pass

        if detector is not None:

            try:
                detector.close()
            except Exception:
                pass

        if camera is not None:

            try:
                camera.stop()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print(
            "System stopped."
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()