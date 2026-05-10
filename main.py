import cv2
from ultralytics import YOLO
from draw_box import *

lap_camera = cv2.VideoCapture(0)
model = YOLO("./detection.pt")

while True:
    success, img = lap_camera.read()
    if success == False:
        continue
    results = model(img, stream = True)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            side_length = x2 - x1
            length = side_length // 6
            
            conf = int(float(box.conf[0]) * 100)
            draw_corners(img, x1,y1,x2,y2,length,2)
            draw_label(img, x1, y1, f"Face: {conf:.2f}",(65,255,0),(0,0,0)) 
            
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    