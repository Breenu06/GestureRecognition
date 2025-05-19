import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('run.mp4')  # Change to your file or 0 for webcam
fgbg = cv2.createBackgroundSubtractorMOG2()
positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 360))
    mask = fgbg.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            positions.append((cx, cy))
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            break

    for i in range(1, len(positions)):
        cv2.line(frame, positions[i-1], positions[i], (255, 0, 0), 2)

    cv2.imshow('Track', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Heatmap
heat = np.zeros((360, 640))
for x, y in positions:
    heat[y, x] += 1
heat = cv2.GaussianBlur(heat, (0, 0), 15)

plt.imshow(heat, cmap='hot')
plt.title('Heatmap')
plt.axis('off')
plt.show()
