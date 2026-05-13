# General Architecture: YoLo -> SORT -> FaceNet -> SVM
YOLO is quite good at detecting objects fast. This might be due to its one-stage architecture. However, it works fine on distinctly different objects, but 
its classification decreases when it has to classify similar objects such as faces. To solve this, we dedicate the classification to another model which 
has better classification performance, especially for faces. But better performance usually comes with the trade-off of speed, from ~60 fps for YOLO down to ~10 fps for YOLO + FaceNet. 
<br/><br/>
To optimize this, we choose a tracking algorithm to assign each object an ID. For each ID, we only need to do the recognition until we have a result score that passes a certain threshold;
then we stop doing recognition. This effectively boosts the speed, achieving a speed almost as fast as detection almost all the time, only slowing down a little when it has to do recognition
for a new ID. The chosen tracking algorithm is SORT (Simple,Online, and Realtime) since it is simple and fast for our real time application.
# Detection: YoLo

# Tracking: SORT

# Recognition: FaceNet and SVM

