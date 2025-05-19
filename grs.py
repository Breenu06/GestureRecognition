# Sign language translation tool
import cv2
import numpy as np

def get_hand_gesture(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is not None:
            count_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = np.linalg.norm(np.array(start) - np.array(end))
                b = np.linalg.norm(np.array(start) - np.array(far))
                c = np.linalg.norm(np.array(end) - np.array(far))
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c + 1e-5))  # Avoid division by zero

                if angle <= np.pi / 2:  # 90 degree
                    count_defects += 1
                    cv2.circle(drawing, far, 5, [0, 0, 255], -1)
            if count_defects >= 4:
                return "Open Hand"
            elif count_defects == 0:
                return "Fist"
            else:
                return "Gesture"
    return "Unknown"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range (adjust if needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours and len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        drawing = np.zeros(roi.shape, np.uint8)
        cv2.drawContours(drawing, [contour], 0, (0, 255, 0), 2)

        gesture = get_hand_gesture(contour, drawing)
        cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display "Hi" when Open Hand is detected
        if gesture == "Open Hand":
            cv2.putText(frame, "Hi", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Resize drawing to fit into frame safely
        drawing_resized = cv2.resize(drawing, (300, 300))
        frame_height, frame_width = frame.shape[:2]

        # Place drawing on frame - adjust based on actual frame width
        if frame_width >= 750:
            frame[100:400, 450:750] = drawing_resized
        else:
            # If frame width is smaller, adjust start position
            frame[100:400, frame_width-300:frame_width] = drawing_resized

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()