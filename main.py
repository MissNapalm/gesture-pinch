import cv2
import mediapipe as mp
import numpy as np
import time

class SimpleHandTracker:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Lower threshold for faster detection
            min_tracking_confidence=0.3,   # Lower threshold for faster tracking
            model_complexity=0
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Cursor settings
        self.cursor_x = 640
        self.cursor_y = 360
        self.smoothing_factor = 0.7
        self.last_cursor_x = self.cursor_x
        self.last_cursor_y = self.cursor_y
        
        # Gesture states
        self.is_pinching = False
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def calculate_distance(self, point1, point2):
        """Calculate distance between two landmark points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def simple_finger_check(self, landmarks, tip_id, pip_id):
        """Check if finger is extended"""
        return landmarks[tip_id].y < landmarks[pip_id].y
    
    def count_extended_fingers(self, landmarks):
        """Count how many fingers are extended"""
        fingers = {
            'thumb': self.simple_finger_check(landmarks, 4, 3),
            'index': self.simple_finger_check(landmarks, 8, 6),
            'middle': self.simple_finger_check(landmarks, 12, 10),
            'ring': self.simple_finger_check(landmarks, 16, 14),
            'pinky': self.simple_finger_check(landmarks, 20, 18)
        }
        return sum(fingers.values()), fingers
    
    def detect_pinch(self, landmarks):
        """Detect if thumb and index finger are pinched together"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < 0.05
    
    def detect_gesture(self, landmarks):
        """Detect finger count gestures"""
        finger_count, fingers = self.count_extended_fingers(landmarks)
        
        # Check for pinch first
        if self.detect_pinch(landmarks):
            return "Pinch"
        
        # Count-based gestures
        if finger_count == 0:
            return "Fist"
        elif finger_count == 1:
            return "One Finger"
        elif finger_count == 2:
            return "Two Fingers"
        elif finger_count == 3:
            return "Three Fingers"
        elif finger_count == 4:
            return "Four Fingers"
        elif finger_count == 5:
            return "Open Hand"
        
        return f"{finger_count} Fingers"
    
    def get_cursor_position(self, landmarks):
        """Get cursor position from hand center"""
        # Use palm center (landmark 9) for cursor position
        palm_center = landmarks[9]
        
        # Get frame dimensions
        h, w = 720, 1280  # Assume standard frame size
        
        # Convert to frame coordinates (no flip - natural movement)
        frame_x = int(palm_center.x * w)
        frame_y = int(palm_center.y * h)
        
        # Apply smoothing
        smooth_x = int(self.smoothing_factor * self.last_cursor_x + (1 - self.smoothing_factor) * frame_x)
        smooth_y = int(self.smoothing_factor * self.last_cursor_y + (1 - self.smoothing_factor) * frame_y)
        
        # Keep in bounds
        smooth_x = max(0, min(smooth_x, w - 1))
        smooth_y = max(0, min(smooth_y, h - 1))
        
        self.last_cursor_x = smooth_x
        self.last_cursor_y = smooth_y
        
        return smooth_x, smooth_y
    
    def draw_cursor(self, frame):
        """Draw cursor at current position"""
        cursor_color = (0, 0, 255) if self.is_pinching else (0, 255, 0)
        cursor_radius = 20 if self.is_pinching else 15
        
        # Main cursor circle
        cv2.circle(frame, (self.cursor_x, self.cursor_y), cursor_radius, cursor_color, -1)
        cv2.circle(frame, (self.cursor_x, self.cursor_y), cursor_radius + 3, cursor_color, 3)
        
        # Crosshair
        line_length = 25
        cv2.line(frame, 
                (self.cursor_x - line_length, self.cursor_y), 
                (self.cursor_x + line_length, self.cursor_y), 
                cursor_color, 3)
        cv2.line(frame, 
                (self.cursor_x, self.cursor_y - line_length), 
                (self.cursor_x, self.cursor_y + line_length), 
                cursor_color, 3)
    
    def draw_info(self, frame):
        """Draw information overlay"""
        h, w = frame.shape[:2]
        
        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Cursor position
        cv2.putText(frame, f"Cursor: ({self.cursor_x}, {self.cursor_y})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Pinch state
        if self.is_pinching:
            cv2.putText(frame, "PINCHING!", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Instructions
        instructions = [
            "Cursor follows hand movement",
            "Pinch thumb+index to activate",
            "Press 'q' to quit"
        ]
        
        start_y = h - len(instructions) * 25 - 10
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, start_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_hand_info(self, frame, hand_data):
        """Draw minimal hand information"""
        for i, (landmarks, hand_label) in enumerate(hand_data):
            # Get hand center for label placement
            hand_center_x = int(landmarks[9].x * frame.shape[1])
            hand_center_y = int(landmarks[9].y * frame.shape[0]) - 40
            
            # Only show pinch status
            is_pinching = self.detect_pinch(landmarks)
            
            # Choose color based on hand
            color = (255, 100, 100) if hand_label == "Right" else (100, 255, 255)
            
            # Only draw pinch status
            if is_pinching:
                label_text = f"{hand_label}: PINCH"
                cv2.putText(frame, label_text, (hand_center_x - 80, hand_center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("🎯 Simple Hand Tracker Started!")
        print("Cursor follows hand movement")
        print("Pinch thumb + index to activate")
        print("Press 'q' to quit")
        
        last_gesture_time = 0
        gesture_cooldown = 1.0  # Longer cooldown to reduce processing
        frame_skip = 0  # For frame skipping optimization
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            frame_skip += 1
            
            # Process with MediaPipe (simplified - removed problematic frame skipping)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            hand_data = []
            self.is_pinching = False
            
            # Process each detected hand
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    hand_label = handedness.classification[0].label
                    hand_data.append((hand_landmarks.landmark, hand_label))
                    
                    # Update cursor position based on hand movement
                    self.cursor_x, self.cursor_y = self.get_cursor_position(hand_landmarks.landmark)
                    
                    # Check for pinch
                    if self.detect_pinch(hand_landmarks.landmark):
                        self.is_pinching = True
                        if not hasattr(self, 'last_pinch_print') or time.time() - self.last_pinch_print > 1.0:
                            print(f"🤏 PINCH detected at ({self.cursor_x}, {self.cursor_y})")
                            self.last_pinch_print = time.time()
            
            # Print gesture information (throttled)
            current_time = time.time()
            if hand_data and current_time - last_gesture_time > gesture_cooldown:
                self.draw_hand_info(frame, hand_data)
                last_gesture_time = current_time
            
            # Draw cursor and info
            self.draw_cursor(frame)
            self.draw_info(frame)
            
            # Calculate FPS
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Show frame
            cv2.imshow('Simple Hand Tracker', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SimpleHandTracker()
    tracker.run()
