# ============================================================
# MOUSE CONTROLLER
# FINAL VERSION
# Compatible with Gesture Controlled System
# ============================================================

import time
import pyautogui


class MouseController:

    def __init__(self, screen_width, screen_height):

        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)

        # Cursor smoothing
        self.prev_x = None
        self.prev_y = None

        self.smoothing = 0.72
        self.dead_zone_px = 1.5

        # Click protection
        self.last_click_time = 0.0
        self.last_right_click_time = 0.0

        self.click_cooldown = 0.45
        self.right_click_cooldown = 0.50

        # Scroll protection
        self.last_scroll_time = 0.0
        self.scroll_cooldown = 0.08

        # PyAutoGUI
        pyautogui.PAUSE = 0

        # Keep FAILSAFE enabled
        pyautogui.FAILSAFE = True


    # ========================================================
    # CLAMP
    # ========================================================

    @staticmethod
    def _clamp(value, minimum, maximum):

        return max(
            minimum,
            min(
                maximum,
                value
            )
        )


    # ========================================================
    # MOVE CURSOR
    # ========================================================

    def move_cursor(
        self,
        x,
        y,
        frame_width,
        frame_height,
        left_margin=0.05,
        right_margin=0.95,
        top_margin=0.06,
        bottom_margin=0.94
    ):

        """
        Move cursor using camera coordinates.

        Compatible with main.py call:

        mouse.move_cursor(
            x,
            y,
            frame_width,
            frame_height,
            left_margin,
            right_margin,
            top_margin,
            bottom_margin
        )
        """

        try:

            frame_width = float(
                frame_width
            )

            frame_height = float(
                frame_height
            )

            if frame_width <= 0:
                return

            if frame_height <= 0:
                return


            # =================================================
            # ACTIVE CAMERA AREA
            # =================================================

            min_x = (
                frame_width
                *
                float(left_margin)
            )

            max_x = (
                frame_width
                *
                float(right_margin)
            )

            min_y = (
                frame_height
                *
                float(top_margin)
            )

            max_y = (
                frame_height
                *
                float(bottom_margin)
            )


            if max_x <= min_x:
                return

            if max_y <= min_y:
                return


            # =================================================
            # CLAMP CAMERA POSITION
            # =================================================

            x = self._clamp(
                float(x),
                min_x,
                max_x
            )

            y = self._clamp(
                float(y),
                min_y,
                max_y
            )


            # =================================================
            # NORMALIZE
            # =================================================

            normalized_x = (
                x - min_x
            ) / (
                max_x - min_x
            )

            normalized_y = (
                y - min_y
            ) / (
                max_y - min_y
            )


            normalized_x = self._clamp(
                normalized_x,
                0.0,
                1.0
            )

            normalized_y = self._clamp(
                normalized_y,
                0.0,
                1.0
            )


            # =================================================
            # SCREEN POSITION
            # =================================================

            target_x = (
                normalized_x
                *
                (self.screen_width - 1)
            )

            target_y = (
                normalized_y
                *
                (self.screen_height - 1)
            )


            # =================================================
            # FIRST POSITION
            # =================================================

            if self.prev_x is None:

                self.prev_x = target_x
                self.prev_y = target_y


            # =================================================
            # SMOOTHING
            # =================================================

            else:

                dx = (
                    target_x
                    -
                    self.prev_x
                )

                dy = (
                    target_y
                    -
                    self.prev_y
                )


                # Small movement = ignore jitter
                if abs(dx) < self.dead_zone_px:

                    dx = 0.0


                if abs(dy) < self.dead_zone_px:

                    dy = 0.0


                self.prev_x += (
                    dx
                    *
                    self.smoothing
                )

                self.prev_y += (
                    dy
                    *
                    self.smoothing
                )


            # =================================================
            # FINAL SCREEN POSITION
            # =================================================

            screen_x = int(
                self._clamp(
                    self.prev_x,
                    0,
                    self.screen_width - 1
                )
            )

            screen_y = int(
                self._clamp(
                    self.prev_y,
                    0,
                    self.screen_height - 1
                )
            )


            # =================================================
            # MOVE
            # =================================================

            pyautogui.moveTo(
                screen_x,
                screen_y,
                duration=0
            )


        except pyautogui.FailSafeException:

            self.reset()


        except Exception as error:

            # Do NOT crash the whole application.
            print(
                "Mouse move warning:",
                error
            )


    # ========================================================
    # LEFT CLICK
    # ========================================================

    def left_click(self):

        now = time.perf_counter()


        if (
            now
            -
            self.last_click_time
            <
            self.click_cooldown
        ):

            return False


        try:

            pyautogui.click()

            self.last_click_time = now

            return True


        except pyautogui.FailSafeException:

            self.reset()

            return False


        except Exception as error:

            print(
                "Left click warning:",
                error
            )

            return False


    # ========================================================
    # RIGHT CLICK
    # ========================================================

    def right_click(self):

        now = time.perf_counter()


        if (
            now
            -
            self.last_right_click_time
            <
            self.right_click_cooldown
        ):

            return False


        try:

            pyautogui.rightClick()

            self.last_right_click_time = now

            return True


        except pyautogui.FailSafeException:

            self.reset()

            return False


        except Exception as error:

            print(
                "Right click warning:",
                error
            )

            return False


    # ========================================================
    # SCROLL
    # ========================================================

    def scroll(
        self,
        amount
    ):

        now = time.perf_counter()


        if (
            now
            -
            self.last_scroll_time
            <
            self.scroll_cooldown
        ):

            return False


        try:

            amount = int(
                amount
            )


            amount = int(
                self._clamp(
                    amount,
                    -6,
                    6
                )
            )


            if amount == 0:

                return False


            pyautogui.scroll(
                amount
            )


            self.last_scroll_time = now

            return True


        except pyautogui.FailSafeException:

            self.reset()

            return False


        except Exception as error:

            print(
                "Scroll warning:",
                error
            )

            return False


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.prev_x = None
        self.prev_y = None