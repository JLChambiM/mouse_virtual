# src/main.py

import cv2
import numpy as np
from hand_detector import HandDetector
from cursor_controller import CursorController

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    hand_detector = HandDetector()
    cursor_controller = CursorController()

    # Configuración de umbrales para gestos
    CLICK_THRESHOLD = 0.03
    GESTURE_CONFIDENCE = 0.85

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame, landmarks = hand_detector.find_hands(frame)

        if landmarks:
            for hand_landmarks in landmarks:
                # Puntos de referencia de los dedos
                index_tip = hand_landmarks.landmark[8]    # Índice
                index_pip = hand_landmarks.landmark[6]    # Segunda articulación del índice
                thumb = hand_landmarks.landmark[4]        # Pulgar
                middle_tip = hand_landmarks.landmark[12]  # Medio
                ring_tip = hand_landmarks.landmark[16]    # Anular

                # Calcular coordenadas de pantalla
                screen_x = int(index_tip.x * cursor_controller.screen_width)
                screen_y = int(index_tip.y * cursor_controller.screen_height)

                # Mover cursor con las mejoras de precisión
                cursor_controller.move_cursor(screen_x, screen_y)

                # Click izquierdo (pulgar + índice)
                click_confidence = cursor_controller.calculate_gesture_confidence(
                    hand_landmarks.landmark,
                    [(4, 8)]
                )
                if click_confidence < CLICK_THRESHOLD:
                    if cursor_controller.check_gesture_stability("click"):
                        cursor_controller.click()
                        cv2.putText(frame, "Left Click!", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Click derecho (pulgar + medio)
                right_click_confidence = cursor_controller.calculate_gesture_confidence(
                    hand_landmarks.landmark,
                    [(4, 12)]
                )
                if right_click_confidence < CLICK_THRESHOLD:
                    if cursor_controller.check_gesture_stability("right_click"):
                        cursor_controller.right_click()
                        cv2.putText(frame, "Right Click!", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Arrastrar y soltar (índice + medio juntos)
                drag_confidence = cursor_controller.calculate_gesture_confidence(
                    hand_landmarks.landmark,
                    [(8, 12)]
                )
                if drag_confidence < CLICK_THRESHOLD:
                    cursor_controller.start_drag()
                    cv2.putText(frame, "Dragging...", (50, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    cursor_controller.stop_drag()

                # Scroll (índice y anular)
                scroll_confidence = cursor_controller.calculate_gesture_confidence(
                    hand_landmarks.landmark,
                    [(8, 16)]
                )
                if scroll_confidence < CLICK_THRESHOLD:
                    scroll_direction = 1 if index_tip.y < index_pip.y else -1
                    cursor_controller.scroll(scroll_direction * 2)
                    cv2.putText(frame, "Scrolling", (50, 130), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Mostrar indicadores de precisión
                cv2.putText(frame, f"Precision: {int((1-click_confidence)*100)}%", 
                          (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Mouse Virtual", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()