import cv2
import mediapipe as mp


class HandDetector:

    def __init__(
        self,
        max_hands=1,
        detection_confidence=0.50,
        tracking_confidence=0.50
    ):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def find_hands(self, frame):

        if frame is None or frame.size == 0:
            return None

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe only reads this image
        rgb.flags.writeable = False

        try:
            results = self.hands.process(rgb)
        except Exception:
            results = None
        finally:
            rgb.flags.writeable = True

        return results

    def close(self):

        if self.hands is not None:
            self.hands.close()
            self.hands = None