import os
import random

import cv2
from ultralytics import YOLO

from tracker import Tracker


vdo_file = os.path.join('.', 'data', 'people.mp4')
vdo_out_file = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(vdo_file)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(vdo_out_file, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

Model = YOLO("yolov8n.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
while ret:

    results = Model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x_1, y_1, x_2, y_2, score, class_id = r
            x_1 = int(x_1)
            x_2 = int(x_2)
            y_1 = int(y_1)
            y_2 = int(y_2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x_1, y_1, x_2, y_2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x_1, y_1, x_2, y_2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (colors[track_id % len(colors)]), 3)

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
