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

# Tracking: SORT

# Recognition: FaceNet and SVM

