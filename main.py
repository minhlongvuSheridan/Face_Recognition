"""
Author: Minh Long Vu
Application: A small application that use SORT tracking to improve the speed of the YOLO + FaceNet/SVM face recognition
Date: 17-05-2026
"""



import threading

import cv2
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from ultralytics import YOLO
from keras_facenet import FaceNet
from draw_box import *
from sort import *
from load_data import *
import time

WHITE = (255,255,255) 
BACKGROUND_GREEN = (0, 60, 0) # Medium Grey for background
LIGHT_GREEN =  (150, 255, 150) # lighter is closer to white so it is INNER
MEDIUM_GREEN = (0, 200, 0) # outer



# ---
TEXT_STRENGTH = 1
TEXT_SIZE = 0.6
TEXT_THICK = 1
BORDER_STRENGTH = 1
BORDER_THICK = 1

names_information = {
    
    "Minh Long Vu":{
        "job": "Student",
        "country": "Canada"
    },
    "Barack Obama":{
        "job": "Former US President",
    "country": "United States"
    },
    "Donal Trump":{
        "job": "US President",
    "country": "United States"
    },
    "Elon Musk":{
        "job": "Owner of SpaceX",
        "country": "United States"
    },
    "ishowspeed":{
        "job": "Streamer",
    "country": "United States"
    },
    "Stephen Chow":{
        "job": "Actor",
        "country": "China"
    }
}





# YOLO model
model = YOLO("./detection.pt") 

# SORT
tracker = Sort(max_age = 20, min_hits = 3,iou_threshold=0.3)
track_dict = dict()

# FaceNet
embedder = FaceNet()
# the path to the images for recognition
dir_path = r'./identity'
train_faces, train_labels = load_dataset_faces(dir_path)
embeded_faces = list()
embeded_faces = embedder.embeddings(train_faces) # Extract features vector
in_encoder = Normalizer(norm='l2') # 
X = in_encoder.transform(embeded_faces) # normalize the feature vectors
out_encoder = LabelEncoder()
out_encoder.fit(train_labels) # learn the unique class and then assign them a numerical value to create mapping
Y = out_encoder.transform(train_labels) # actually perform the conversion based on what it learn before

# SVM model
SVM_model = SVC(kernel='linear', probability=True)
SVM_model.fit(X, Y)

embeded_images_id = [True, dict()]
def recognition_threading(to_embbeded_images, to_embedded_id):
    embedded_images = embedder.embeddings(to_embbeded_images)
    for i, image in enumerate(embedded_images):
        id = to_embedded_id[i]
        embeded_images_id[1][id] = image
    embeded_images_id[0] = True



lap_camera = cv2.VideoCapture(0)
while True:
    start_time = time.time()
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
            if conf >= 50:
                currentArray = np.array([x1, y1, x2, y2, conf])
                Detections = np.vstack((Detections, currentArray))
            
    tracker_results = tracker.update(Detections)
    to_embbeded_images = []
    to_embbeded_id = []
    for tracker_result in tracker_results:
        x1, y1, x2, y2, id = map(int, tracker_result)
        # if negative then convert it to zero
        x1, y1, x2, y2 = [max(0, i) for i in (x1, y1, x2, y2)]
        if id in track_dict:
            if(track_dict[id][0] == False):
                name = track_dict.get(id)[3]
                name_info= dict()
                name_info["name"] = name
                print(name)
                name_info["job"] = names_information[name]["job"]
                name_info["country"] = names_information[name]["country"]
                draw_neon_corners(img, x1, y1, x2, y2,
                                LIGHT_GREEN, MEDIUM_GREEN, BORDER_STRENGTH, length, BORDER_THICK)
                
                draw_neon_label(img, x1, y1, x2, y2, name_info, 
                                LIGHT_GREEN, MEDIUM_GREEN, TEXT_SIZE,
                                BACKGROUND_GREEN, LIGHT_GREEN, MEDIUM_GREEN, BORDER_STRENGTH,BORDER_THICK)
        else:
            point1 = (x1,x2)
            point2 = (y1,y2)
            track_dict[id] = [True, point1, point2, "Unknow"]
            sub_image = img.copy()[y1:y2,x1:x2]
            to_embbeded_images.append(sub_image)
            to_embbeded_id.append((id))
    
    
    if embeded_images_id[0] == True:
        for key, value in embeded_images_id[1].items():
            id = key
            embbeded_image = value
            normalized_sub_feat = in_encoder.transform([embbeded_image])
            preds = SVM_model.predict_proba(normalized_sub_feat)
            best_class_idx = np.argmax(preds[0])
            probability = preds[0][best_class_idx] * 100
    
            if(probability >= 50):
                name = out_encoder.inverse_transform([best_class_idx])[0]
                track_dict[id][0] = False
                track_dict[id][3] = name
            else: 
                to_embbeded_images.append(sub_image)
                to_embbeded_id.append((id))
        embeded_images_id[0] = False
        embeded_images_id[1].clear()
            
            
            
    # we don't want to change the size of embeded_images
    if len(to_embbeded_images) > 0 and embeded_images_id[0] == False:
        embeded_images_id[0] = False
        threading.Thread(target=recognition_threading, args=(to_embbeded_images,to_embbeded_id,)).start()

    end_time = time.time()
    elapsed_time = end_time - start_time  
    cv2.putText(img, f"{1 // elapsed_time} FPS",(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 50, 32), 1)      
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    