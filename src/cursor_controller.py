# src/cursor_controller.py

import pyautogui
import numpy as np

class CursorController:
    def __init__(self):
        pyautogui.FAILSAFE = True
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Parámetros de suavizado y precisión
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 0.7
        self.movement_buffer = []
        self.buffer_size = 5
        
        # Zona muerta y límites
        self.deadzone = 20
        self.border_margin = 10
        self.is_dragging = False
        
        # Historial para gestos
        self.gesture_history = []
        self.gesture_buffer_size = 3

    def apply_smoothing(self, x, y):
        self.movement_buffer.append((x, y))
        if len(self.movement_buffer) > self.buffer_size:
            self.movement_buffer.pop(0)
        
        if len(self.movement_buffer) == self.buffer_size:
            x = sum(p[0] for p in self.movement_buffer) / self.buffer_size
            y = sum(p[1] for p in self.movement_buffer) / self.buffer_size

        smoothed_x = int(self.smoothing * x + (1 - self.smoothing) * self.prev_x)
        smoothed_y = int(self.smoothing * y + (1 - self.smoothing) * self.prev_y)
        
        return smoothed_x, smoothed_y

    def apply_deadzone(self, x, y):
        if abs(x - self.prev_x) < self.deadzone and abs(y - self.prev_y) < self.deadzone:
            return self.prev_x, self.prev_y
        return x, y

    def constrain_to_screen(self, x, y):
        x = max(self.border_margin, min(self.screen_width - self.border_margin, x))
        y = max(self.border_margin, min(self.screen_height - self.border_margin, y))
        return x, y

    def move_cursor(self, x, y):
        x, y = self.apply_smoothing(x, y)
        x, y = self.apply_deadzone(x, y)
        x, y = self.constrain_to_screen(x, y)
        self.prev_x, self.prev_y = x, y
        pyautogui.moveTo(x, y)

    def check_gesture_stability(self, gesture_type):
        self.gesture_history.append(gesture_type)
        if len(self.gesture_history) > self.gesture_buffer_size:
            self.gesture_history.pop(0)
        
        if len(self.gesture_history) == self.gesture_buffer_size:
            return all(g == gesture_type for g in self.gesture_history)
        return False

    def calculate_gesture_confidence(self, landmarks, gesture_points):
        distances = []
        for p1, p2 in gesture_points:
            dist = np.sqrt((landmarks[p1].x - landmarks[p2].x)**2 + 
                         (landmarks[p1].y - landmarks[p2].y)**2)
            distances.append(dist)
        return np.mean(distances)

    # Agregando los métodos faltantes
    def click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.rightClick()

    def start_drag(self):
        if not self.is_dragging:
            pyautogui.mouseDown()
            self.is_dragging = True

    def stop_drag(self):
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False

    def scroll(self, distance):
        pyautogui.scroll(distance)