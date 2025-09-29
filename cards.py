import cv2
import mediapipe as mp
import numpy as np
import math

# Constants
CARD_COUNT = 7
CARD_WIDTH = 420
CARD_HEIGHT = 400
CARD_SPACING = 60

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandState:
    def __init__(self):
        self.mode = 'gesture'  # 'gesture' or 'card'
        self.card_index = 0
        self.target_index = 0
        self.card_anim_pos = 0.0
        self.last_ok_time = 0
        self.ok_gesture_count = 0
        self.last_scroll = 0
        self.last_scroll_time = 0

# Utility: Check if hand is doing OK gesture (thumb tip and index tip close)
def is_ok_gesture(landmarks):
    if not landmarks:
        return False
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return dist < 0.06


# Utility: Check if hand is doing three-finger gesture (index, middle, ring extended)
def is_three_finger_gesture(landmarks):
    if not landmarks:
        return False
    def extended(tip, pip):
        return landmarks[tip].y < landmarks[pip].y
    return (
        extended(8, 6) and extended(12, 10) and extended(16, 14)
        and not extended(20, 18)
    )

# Utility: Check if hand is doing one-finger point gesture (index finger extended)
def is_one_finger_point(landmarks):
    if not landmarks:
        return False
    def extended(tip, pip):
        return landmarks[tip].y < landmarks[pip].y
    return (
        extended(8, 6)
        and not extended(12, 10)
        and not extended(16, 14)
        and not extended(20, 18)
    )

def draw_cards(frame, center_x, center_y, anim_pos):
    for i in range(CARD_COUNT):
        offset = (i - anim_pos) * (CARD_WIDTH + CARD_SPACING)
        x = int(center_x + offset)
        y = int(center_y)
        color = (255, 255, 255) if round(anim_pos) == i else (200, 200, 200)
        cv2.rectangle(frame, (x - CARD_WIDTH//2, y - CARD_HEIGHT//2),
                      (x + CARD_WIDTH//2, y + CARD_HEIGHT//2), color, -1)
        cv2.rectangle(frame, (x - CARD_WIDTH//2, y - CARD_HEIGHT//2),
                      (x + CARD_WIDTH//2, y + CARD_HEIGHT//2), (0,0,0), 4)
        # Center the text on the card
        text = f"App {i+1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0,0,0), thickness)

def main():
    import time
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    state = HandState()
    last_scroll_hand = None
    last_scroll_time = 0
    ANIM_SPEED = 0.18  # Lower is slower
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_landmarks = [h.landmark for h in results.multi_hand_landmarks] if results.multi_hand_landmarks else []

        # Reset if no hands
        if not hand_landmarks:
            state.mode = 'gesture'
            state.ok_gesture_count = 0
            state.card_index = 0
            state.target_index = 0
            state.card_anim_pos = float(state.card_index)
        # --- GESTURE MODE ---
        elif state.mode == 'gesture':
            ok_count = sum(is_ok_gesture(lm) for lm in hand_landmarks)
            if ok_count >= 2:
                state.ok_gesture_count += 1
            else:
                state.ok_gesture_count = 0
            if state.ok_gesture_count > 5:
                state.mode = 'card'
                state.ok_gesture_count = 0
                state.target_index = state.card_index
                state.card_anim_pos = float(state.card_index)
        # --- CARD VIEW MODE ---
        elif state.mode == 'card':
            now = time.time()
            scrolled = False
            for idx, hand in enumerate(hand_landmarks):
                # Three-finger gesture: left/right
                if is_three_finger_gesture(hand):
                    hand_x = hand[0].x
                    if hand_x < 0.5:
                        # Left hand: scroll left
                        if state.target_index > 0 and (last_scroll_hand != 'left' or now - last_scroll_time > 0.5):
                            state.target_index -= 1
                            last_scroll_hand = 'left'
                            last_scroll_time = now
                            scrolled = True
                    else:
                        # Right hand: scroll right
                        if state.target_index < CARD_COUNT - 1 and (last_scroll_hand != 'right' or now - last_scroll_time > 0.5):
                            state.target_index += 1
                            last_scroll_hand = 'right'
                            last_scroll_time = now
                            scrolled = True
                # One-finger point: left/right
                elif is_one_finger_point(hand):
                    hand_x = hand[0].x
                    if hand_x < 0.5:
                        # Left hand: scroll left
                        if state.target_index > 0 and (last_scroll_hand != 'left' or now - last_scroll_time > 0.5):
                            state.target_index -= 1
                            last_scroll_hand = 'left'
                            last_scroll_time = now
                            scrolled = True
                    else:
                        # Right hand: scroll right
                        if state.target_index < CARD_COUNT - 1 and (last_scroll_hand != 'right' or now - last_scroll_time > 0.5):
                            state.target_index += 1
                            last_scroll_hand = 'right'
                            last_scroll_time = now
                            scrolled = True
            if not scrolled:
                last_scroll_hand = None
            # Animate card position
            state.card_anim_pos += (state.target_index - state.card_anim_pos) * ANIM_SPEED
            # Snap if close
            if abs(state.card_anim_pos - state.target_index) < 0.01:
                state.card_anim_pos = float(state.target_index)

        # Draw UI
        if state.mode == 'card':
            draw_cards(frame, w//2, h//2, state.card_anim_pos)
            cv2.putText(frame, "Select App", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)
        else:
            cv2.putText(frame, "Gesture Mode: Show double OK to enter Card View", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Hand Carousel', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
