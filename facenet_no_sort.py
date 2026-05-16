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
GREY = (128, 128, 128) # Medium Grey for background
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
    start_time = time.time()
    success, img = lap_camera.read()
    if success == False:
        print("Fail to read the next frame")
        continue
    Detections = np.empty((0, 5))
    results = model(img, stream = True)
    to_embbeded_images = []
    to_embbeded_metadata = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            side_length = x2 - x1
            length = side_length // 6
            
            conf = int(float(box.conf[0]) * 100)
            
            if conf >= 50:
                sub_image = img.copy()[y1:y2,x1:x2]
                to_embbeded_images.append(sub_image)
                to_embbeded_metadata.append((x1, y1, x2, y2))
    if len(to_embbeded_images) > 0:
        embedded_images = embedder.embeddings(to_embbeded_images)
        for i, embeded_image in enumerate(embedded_images): 
            (x1, y1, x2, y2) = to_embbeded_metadata[i]
            normalized_sub_feat = in_encoder.transform([embeded_image])
            preds = SVM_model.predict_proba(normalized_sub_feat)
            best_class_idx = np.argmax(preds[0])
            probability = preds[0][best_class_idx] * 100
            name = out_encoder.inverse_transform([best_class_idx])[0]
            name_info= dict()
            name_info["name"] = name
            print(name)
            name_info["job"] = names_information[name]["job"]
            name_info["country"] = names_information[name]["country"]
            draw_neon_corners(img, x1, y1, x2, y2,
                                LIGHT_GREEN, MEDIUM_GREEN, BORDER_STRENGTH, length, BORDER_THICK)
            draw_neon_label(img, x1, y1, x2, y2, name_info, 
                            LIGHT_GREEN, MEDIUM_GREEN,TEXT_STRENGTH, TEXT_SIZE,
                            GREY, LIGHT_GREEN, MEDIUM_GREEN, BORDER_STRENGTH,BORDER_THICK)
    
    end_time = time.time()
    elapsed_time = end_time - start_time  
    cv2.putText(img, f"{1 // elapsed_time} FPS",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 50, 32), 1)      
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    