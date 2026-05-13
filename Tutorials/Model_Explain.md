# General Architecture: YoLo -> SORT -> FaceNet -> SVM
YOLO is quite good at detecting objects fast. This might be due to its one-stage architecture.
<img width="959" height="72" alt="image" src="https://github.com/user-attachments/assets/f0146f62-5223-4c28-855a-e0f2389e91b1" /><br/>
Figure 1. Time needed to detect 3 Faces
<img width="918" height="75" alt="image" src="https://github.com/user-attachments/assets/30e7dcba-7425-4612-ab83-0ee93ffb2177" /><br/>
Figure 2. Time needed to detect 3 Faces <br/>
As see in figure 1 and 2. There are no different between detecting 3 faces or 1 faces<br/>
It works fine on distinctly different objects, but its classification decreases when it has to classify similar objects such as faces. To solve this, we dedicate the classification to another model which has better classification performance, especially for faces. But better performance usually comes with the trade-off of speed, from ~25 fps for YOLO down to ~10 fps for YOLO + FaceNet. 

https://github.com/user-attachments/assets/187d0208-665d-49c2-bdf2-5ca05152205a

Video 1. YoLO + FaceNet  
As seen above, the speed is incredibly slow with the fps drop to 5-6 fps when it detected a face. It is even worse when one more face is detected, where it drop to 3 fps. The main reason 
is because it take around 60ms to perform an recognition for an object. If we have 2 objects, it would be 120ms just for recognition part.<br/>
<img width="970" height="145" alt="image" src="https://github.com/user-attachments/assets/c924332d-4607-426b-a8bf-ccff3d91f9bc" /><br/>
Figure 3. Time need to do a recognition for an object
To optimize this, we choose a tracking algorithm to assign each object an ID. For each ID, we only need to do the recognition until we have a result score that passes a certain threshold;
then we stop doing recognition. This effectively boosts the speed, achieving a speed almost as fast as detection almost all the time, only slowing down a little when it has to do recognition
for a new ID. The chosen tracking algorithm is SORT (Simple,Online, and Realtime) since it is simple and fast for our real time application.

https://github.com/user-attachments/assets/d4823398-54f2-4133-9fea-dfcb379b8587

Video 2. YoLo + SORT + FaceNet

Video 2 showns that it is real time almost all the time, the FPS only drops when it detect new face, however since it is just a small time so the overall experience is not significantly
affected.

# Detection: YoLo
The job of YOLO detection is to answer the question "Where is the face". <br/>
The YoLo is an deep learning model for object detection so it can not automatically process the image. Thus, we need to use OpenCv for computer vision task to feed the image to it.
- Step 1: We open the camera of laptop.
  ```python
  lap_camera = cv2.VideoCapture(0)
  ```
- Step 2: Load YOLO model. Please watch *Train_Yolo.md* to see how to train a Yolo model
  ```python
  model = YOLO("./detection.pt") 
  ```
  Step 3: Continuosly capture the image frames and show it
  ```python
  while True:
    success, img = lap_camera.read()
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
  ```
  Step 4: Run the model on the captured frame
  ```python
  results = model(img, stream = True)
  ```
  Step 5: Extract the bounding box and confidence score <br/>
  right below *lap_camera.read* add this
  ```python
  for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = int(float(box.conf[0]) * 100)
  ```
  (x1,y1) and (x2,y2) are the top left and bottom right corners of bounding box, respectively. Internally they use tensor type so
  we need to convert them back to integer to use
  Step 6: Draw the customized bounding box and labels on the frame before displaying it
  ```python
  side_length = x2 - x1
  length = side_length // 6
  draw_corners(img, x1,y1,x2,y2,length,2)
  draw_label(img, x1, y1, f"Face: {conf}",(65,255,0),(0,0,0)) 
  ```
  *draw_label* and *draw_corners* are from the *draw_box.py*
The complete code for detection 
```python
import cv2
from ultralytics import YOLO
from draw_box import *

model = YOLO("./detection.pt") 
lap_camera = cv2.VideoCapture(0)

while True:
    success, img = lap_camera.read()
    results = model(img, stream = True)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = int(float(box.conf[0]) * 100)
            side_length = x2 - x1
            length = side_length // 6
            draw_corners(img, x1,y1,x2,y2,length,2)
            draw_label(img, x1, y1, f"Face: {conf}",(65,255,0),(0,0,0)) 
    cv2.imshow("Image", img)
    cv2.waitKey(1)
```
You should be able to run this code before progressing to the next step
# Tracking: SORT
When you already known "Where is the face". The next question you might ask "Is this the same face as previous one". Answering that question is the job of tracking. The Simple,Online and Realtime - SORT algorithm provide a simple tracking. It is simple because it only consider the the Intersect-Over-Union IoU <br/>
<img width="453" height="262" alt="image" src="https://github.com/user-attachments/assets/0215d3c0-8a98-4fe6-8b99-b23e97447ae8" /><br/>
it is fast due to its minimalistic implementation of tracking. It fail to track for the same object reenter again. Furthermore, it could mistake tracking if two similiar objects are near to eacho ther. Since our project focus more on real time, we choose this simple algorithm<br/>

Step 1: Create Tracking object and dictionary<br/>
Add this below *model = YOLO("./detection.pt")*
```python
tracker = Sort(max_age = 20, min_hits = 3,iou_threshold=0.3)
track_dict = dict()
```
the *track_dict* is used to stored id and the identity associated with that id
Step 2: Create Detection array to store information of detected objects want to track<br/>
Add this before *results = model(img, stream = True)*
```
Detections = np.empty((0, 5))
```
*(0,5)* because that is array they work internally
- Step 3: Conditionally add the object to detection array<br/>
Add this below *conf = int(float(box.conf[0]) * 100)*
```python
if conf > 0.5:
  currentArray = np.array([x1, y1, x2, y2, conf])
  Detections = np.vstack((Detections, currentArray))
```
- Step 4: run the tracker on the Detection array <br/>
Add this outside below the the while loop *for result in results:*
```
tracker_results = tracker.update(Detections)
```
The tracker receive array object with the format [x1,y1,x2,y2,conf] and return [x1,y1,x2,y2,id]
This id is what we use to track the object
# Recognition: FaceNet and SVM
After knowing that if it is the same face, we will ask final key question "whose face is this". This type of question is called Recognition. 
The FaceNet is not a classifier. It is just a model to extract the vector of features of an object. This is what we called "embedding" step. What to do with this feature is up to us.
However, we couldn't conclude anything with just feature vector. The general idea is to apply Machine Learning techniques such as KNN, decision Tree,SVM... In that I used Support Vector
Machine or SVM.<br/>
SVM works by define the two support vector point and then seperate the space between those vectors. Better SVM would have the better marginal distance.
<img width="623" height="458" alt="image" src="https://github.com/user-attachments/assets/07447244-67c3-463e-820c-aec9eacf6b56" />









