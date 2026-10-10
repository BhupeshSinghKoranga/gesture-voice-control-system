import os
import warnings

import joblib
import numpy as np


EXPECTED_FEATURE_COUNT = 42


class MLPredictor:

    def __init__(
        self,
        model_path="models/gesture_model.pkl"
    ):

        candidates = [
            model_path,
            os.path.join(
                "models",
                "gesture_model.pkl"
            ),
            os.path.join(
                "app",
                "models",
                "gesture_model.pkl"
            )
        ]

        self.model_path = next(
            (
                path
                for path in candidates
                if os.path.isfile(path)
            ),
            None
        )

        if self.model_path is None:

            raise FileNotFoundError(
                "gesture_model.pkl not found.\n\n"
                "Put the model inside:\n"
                "models/gesture_model.pkl\n"
                "or\n"
                "app/models/gesture_model.pkl"
            )

        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names"
        )

        try:

            self.model = joblib.load(
                self.model_path
            )

        except Exception as exc:

            raise RuntimeError(
                f"Could not load model:\n"
                f"{self.model_path}\n\n"
                f"Reason: {exc}"
            ) from exc

        if not hasattr(
            self.model,
            "predict"
        ):

            raise TypeError(
                "Loaded model is not a valid "
                "classification model."
            )

        self.feature_count = (
            EXPECTED_FEATURE_COUNT
        )

    def extract_features(
        self,
        hand_landmarks
    ):

        if hand_landmarks is None:

            return np.empty(
                0,
                dtype=np.float32
            )

        landmarks = getattr(
            hand_landmarks,
            "landmark",
            None
        )

        if (
            landmarks is None
            or
            len(landmarks) != 21
        ):

            return np.empty(
                0,
                dtype=np.float32
            )

        points = np.array(
            [
                [
                    float(lm.x),
                    float(lm.y)
                ]
                for lm in landmarks
            ],
            dtype=np.float32
        )

        if not np.isfinite(
            points
        ).all():

            return np.empty(
                0,
                dtype=np.float32
            )

        # Wrist = origin
        points -= points[0].copy()

        # Scale normalization
        distances = np.linalg.norm(
            points,
            axis=1
        )

        max_distance = float(
            np.max(distances)
        )

        if max_distance <= 1e-8:

            return np.empty(
                0,
                dtype=np.float32
            )

        points /= max_distance

        features = points.flatten()

        if len(features) != 42:

            return np.empty(
                0,
                dtype=np.float32
            )

        return features.astype(
            np.float32,
            copy=False
        )

    def predict(
        self,
        hand_landmarks
    ):

        features = self.extract_features(
            hand_landmarks
        )

        if len(features) != 42:

            return (
                "UNKNOWN",
                0.0
            )

        model_input = features.reshape(
            1,
            -1
        )

        try:

            prediction = (
                self.model.predict(
                    model_input
                )[0]
            )

        except Exception:

            return (
                "UNKNOWN",
                0.0
            )

        confidence = 0.0

        if hasattr(
            self.model,
            "predict_proba"
        ):

            try:

                probabilities = (
                    self.model.predict_proba(
                        model_input
                    )[0]
                )

                if len(probabilities):

                    confidence = float(
                        np.max(
                            probabilities
                        )
                    )

            except Exception:

                confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        return (
            str(prediction),
            confidence
        )