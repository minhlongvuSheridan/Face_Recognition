import cv2
from ultralytics import YOLO
from draw_box import *
from sort import *

lap_camera = cv2.VideoCapture(0)
model = YOLO("./detection.pt")

tracker = Sort(max_age = 20, min_hits = 3,iou_threshold=0.3)


while True:
    
    success, img = lap_camera.read()
    if success == False:
        continue
    Detections = np.empty((0, 5))
    results = model(img, stream = True)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            side_length = x2 - x1
            length = side_length // 6
            
            conf = int(float(box.conf[0]) * 100)
            currentArray = np.array([x1, y1, x2, y2, conf])
            Detections = np.vstack((Detections, currentArray))
            
    tracker_results = tracker.update(Detections)
    for tracker_result in tracker_results:
        x1, y1, x2, y2, id = map(int, tracker_result)
        # if negative then convert it to zero
        x1, y1, x2, y2 = [max(0, i) for i in (x1, y1, x2, y2)]
        draw_corners(img, x1,y1,x2,y2,length,2)
        draw_label(img, x1, y1, f"Face - {id}",(65,255,0),(0,0,0)) 
            
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    