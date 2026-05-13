import cv2
from os import listdir

def load_dataset_faces(directory):
    faces = list()
    names = list()
    for filename in listdir(directory):
        
        path = directory + '\\'+ filename
        name = filename
        for img in listdir(path):
            face = cv2.imread(path + '\\' + img)
            faces.append(face)
            names.append(name)
    return faces, names