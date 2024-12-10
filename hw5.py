import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Camera and screen parameters
camera_width = 640
camera_height = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
screen_width, screen_height = pyautogui.size()

# Utility functions
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def angle_with_vertical(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length == 0:
        return 90.0
    cos_theta = dy / length
    theta = math.degrees(math.acos(cos_theta))
    return theta

# Adjusted thresholds
pinch_threshold = 0.05
thumbs_up_threshold = 0.08       # Increased from 0.05 to avoid false positives
thumbs_down_threshold = 0.12     # Increased from 0.15
vertical_angle_threshold = 15.0

# Scrolling gesture thresholds
scroll_pinch_threshold = 0.05
scroll_confirm_frames = 5

# Gesture detection frame requirements
pinch_confirm_frames = 3
thumbs_up_confirm_frames = 5
thumbs_down_confirm_frames = 5

# Frame counters
pinch_detected_frames = 0
thumbs_up_detected_frames = 0
thumbs_down_detected_frames = 0
scroll_detected_frames = 0

# Keep a short history of wrist positions for smoothing
position_history = deque(maxlen=5)

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while True:
        success, image = cap.read()
        if not success:
            break

        # Flip image for a selfie view
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find hand landmarks
        results = hands.process(rgb_image)

        # Clear frame detections
        frame_pinch_detected = False
        frame_thumbs_up_detected = False
        frame_thumbs_down_detected = False
        frame_scroll_detected = False
        frame_scroll_direction = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract relevant landmarks
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                wrist = landmarks[0]
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                index_mcp = landmarks[5]
                thumb_mcp = landmarks[2]
                middle_tip = landmarks[12]

                # Calculate pinch distances
                click_pinch_dist = distance(thumb_tip, index_tip)   # For clicking
                scroll_pinch_dist = distance(thumb_tip, middle_tip) # For scrolling

                # Detect click pinch first
                if click_pinch_dist < pinch_threshold:
                    frame_pinch_detected = True

                # Detect scroll pinch only if no click pinch is present
                if not frame_pinch_detected and (scroll_pinch_dist < scroll_pinch_threshold):
                    frame_scroll_detected = True
                    # Decide scroll direction based on middle fingertip relative to wrist
                    if middle_tip[1] < wrist[1]:
                        frame_scroll_direction = 'up'
                    else:
                        frame_scroll_direction = 'down'

                # Only check thumbs-up/down if no pinch gestures are active
                # This reduces conflicts when moving thumb towards the middle finger.
                if not frame_pinch_detected and not frame_scroll_detected:
                    vertical_angle = angle_with_vertical(thumb_mcp, thumb_tip)

                    # Thumbs up: thumb tip well above index MCP
                    # Increase threshold to reduce accidental triggers
                    if (index_mcp[1] - thumb_tip[1]) > thumbs_up_threshold:
                        # Additional check: Ensure thumb isn't near middle finger tip, 
                        # reducing confusion with scroll pinch
                        if distance(thumb_tip, middle_tip) > scroll_pinch_threshold * 1.5:
                            frame_thumbs_up_detected = True

                    # Thumbs down: thumb tip well below index MCP and nearly vertical
                    if ((thumb_tip[1] - index_mcp[1]) > thumbs_down_threshold) and (vertical_angle < vertical_angle_threshold):
                        # Also check that we're not accidentally close to index tip (click pinch)
                        # or middle tip (scroll pinch)
                        if distance(thumb_tip, index_tip) > pinch_threshold * 1.5 and \
                           distance(thumb_tip, middle_tip) > scroll_pinch_threshold * 1.5:
                            frame_thumbs_down_detected = True

                # Add wrist position to history
                position_history.append(wrist)
                avg_wrist_x = np.mean([pos[0] for pos in position_history])
                avg_wrist_y = np.mean([pos[1] for pos in position_history])

                # Move cursor based on wrist position
                cursor_x = int(avg_wrist_x * screen_width)
                cursor_y = int(avg_wrist_y * screen_height)
                pyautogui.moveTo(cursor_x, cursor_y, duration=0.0)

        # Handle pinch-based click logic
        if frame_pinch_detected:
            pinch_detected_frames += 1
        else:
            pinch_detected_frames = 0

        if pinch_detected_frames == pinch_confirm_frames:
            pyautogui.click()
            pinch_detected_frames = 0

        # Handle thumbs-up volume increase logic
        if frame_thumbs_up_detected:
            thumbs_up_detected_frames += 1
        else:
            thumbs_up_detected_frames = 0

        if thumbs_up_detected_frames == thumbs_up_confirm_frames:
            for _ in range(2):
                pyautogui.press("volumeup")
            thumbs_up_detected_frames = 0

        # Handle thumbs-down volume decrease logic
        if frame_thumbs_down_detected:
            thumbs_down_detected_frames += 1
        else:
            thumbs_down_detected_frames = 0

        if thumbs_down_detected_frames == thumbs_down_confirm_frames:
            for _ in range(2):
                pyautogui.press("volumedown")
            thumbs_down_detected_frames = 0

        # Handle scroll gesture logic
        if frame_scroll_detected:
            scroll_detected_frames += 1
        else:
            scroll_detected_frames = 0

        if scroll_detected_frames == scroll_confirm_frames:
            if frame_scroll_direction == 'up':
                pyautogui.scroll(200)
            else:
                pyautogui.scroll(-200)
            scroll_detected_frames = 0

        cv2.imshow("Gesture Control", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
