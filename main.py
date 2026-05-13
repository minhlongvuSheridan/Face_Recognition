import cv2
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from ultralytics import YOLO
from keras_facenet import FaceNet
from draw_box import *
from sort import *
from load_data import *



# YOLO model
model = YOLO("./detection.pt") 

# SORT
tracker = Sort(max_age = 20, min_hits = 3,iou_threshold=0.3)
track_dict = dict()

# FaceNet
embedder = FaceNet()
# the path to the images for recognition
dir_path = r'C:\Users\haivu\OneDrive - Sheridan College\Desktop\computer\Project\Face_Recognition\identity'
train_faces, train_labels = load_dataset_faces(dir_path)
embeded_faces = list()
embeded_faces = embedder.embeddings(train_faces) # Extract features vector
in_encoder = Normalizer(norm='l2') # 
X = in_encoder.transform(embeded_faces) # normalize the feature vectors
out_encoder = LabelEncoder()
out_encoder.fit(train_labels) # encode the string to proper format for SVM
Y = out_encoder.transform(train_labels)

# SVM model
SVM_model = SVC(kernel='linear', probability=True)
SVM_model.fit(X, Y)


lap_camera = cv2.VideoCapture(0)
while True:
    
    success, img = lap_camera.read()
    if success == False:
        print("Fail to read the next frame")
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
        if id in track_dict:
            name = track_dict.get(id)[0]
            probability = track_dict.get(id)[1]
            draw_corners(img, x1,y1,x2,y2,length,2)
            draw_label(img, x1, y1, f"{name}: {probability:.2f}",(65,255,0),(0,0,0)) 
        else:
            sub_image = img.copy()[y1:y2,x1:x2]
            sub_feat = embedder.embeddings([sub_image])
            sub_feat_norm = in_encoder.transform(sub_feat)
            preds = SVM_model.predict_proba(sub_feat_norm)
            best_class_idx = np.argmax(preds)
            probability = preds[0][best_class_idx] * 100
            name = out_encoder.inverse_transform([best_class_idx])[0]
            if(probability >= 60):
                track_dict[id] = [name, probability]
                draw_corners(img, x1,y1,x2,y2,length,2)
                draw_label(img, x1, y1, f"{name}: {probability:.2f}",(65,255,0),(0,0,0))
    
            
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    