import math


class GestureClassifier:
    """
    Classifies basic hand gestures using MediaPipe landmarks.

    Landmark IDs:
    0  = Wrist
    4  = Thumb tip
    5  = Index MCP
    6  = Index PIP
    8  = Index tip
    9  = Middle MCP
    10 = Middle PIP
    12 = Middle tip
    13 = Ring MCP
    14 = Ring PIP
    16 = Ring tip
    17 = Pinky MCP
    18 = Pinky PIP
    20 = Pinky tip
    """

    def _distance(self, p1, p2):
        return math.hypot(p1[1] - p2[1], p1[2] - p2[2])

    def _finger_up(self, landmarks, tip_id, pip_id):
        return landmarks[tip_id][2] < landmarks[pip_id][2]

    def classify(self, landmarks):
        if len(landmarks) != 21:
            return "UNKNOWN", 0.0

        # Finger states
        index_up = self._finger_up(landmarks, 8, 6)
        middle_up = self._finger_up(landmarks, 12, 10)
        ring_up = self._finger_up(landmarks, 16, 14)
        pinky_up = self._finger_up(landmarks, 20, 18)

        # Pinch distance between thumb and index finger
        pinch_distance = self._distance(
            landmarks[4],
            landmarks[8]
        )

        # Palm size for scale-independent pinch detection
        palm_size = self._distance(
            landmarks[0],
            landmarks[9]
        )

        normalized_pinch = pinch_distance / max(palm_size, 1)

        # 1. PINCH
        if normalized_pinch < 0.45:
            return "PINCH", 0.95

        # 2. FIST / EMERGENCY STOP
        if not index_up and not middle_up and not ring_up and not pinky_up:
            return "FIST", 0.95

        # 3. TWO FINGERS
        if index_up and middle_up and not ring_up and not pinky_up:
            return "TWO_FINGERS", 0.90

        # 4. INDEX FINGER
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "INDEX", 0.92

        # 5. OPEN PALM
        if index_up and middle_up and ring_up and pinky_up:
            return "OPEN_PALM", 0.95

        return "UNKNOWN", 0.50