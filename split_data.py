'''
Author: Minh Long Vu

'''

import os
from os import listdir
import shutil
from sklearn.model_selection import train_test_split
def split_data(images_dir,labels_dir, percent_val=0.15,percent_test=0.15):
    
    '''
    This split the data into train, val, and test folders. Each contain images and labels folders
    It assume that the label files has the same name as images files Example: img1.jpg and img1.txt
    images_src: the source of the images
    labels_src: the source of the labels
    percent_train: the percentarge of the train part
    percent_val: the percentage of validation part
    percent_test: the percentage of test part
    return: return True if successful
    '''
    images = list()
    for img in listdir(images_dir):
        # only take the name not extension
        images.append(img.split(".")[0])

    '''
    we will perform two splits
    First split of images: train_val and test parts
    Second split of train_val: train and val
    First split we only specify the train_val with 80 but
    what is the parameter for second split?
    
    Assume that N is the total rows where train,val, and test 
    are the parts with rows corresponding to percentage of total rows
    train + val + test = N
    Where
    (1) train = 0.7N, val = 0.15N and test = 0.15N
    After first split where we extract the test portion
    we have
    train + val = 0.85N
    The question is that how much percentage of these 85N we need to take
    so that it could be equal with the inital percentage of total rows. 
    We call those portions x and y for train and val, respectively
    (2) val = x * 85N, val = y * 15N
    From (1) and (2) we have
    val = y * 0.85N= 0.15N => x = 0.15 / 0.85 ~~ 0,1764 ~~ 0,18
    '''
    
    # First split
    train_val, test, _, _ = train_test_split(
    images, images, 
    test_size=percent_test, 
    random_state=42
    )
    val_size = percent_val / (1 - percent_test)
    train, val, _, _ = train_test_split(
    train_val, train_val, 
    test_size= val_size, 
    random_state=42
    )
    return train, val, test

def copy_images_to_dest(src_dir, dest_dir, src_files):
    # there might be mess up between jpg and png 
    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
    for fl in src_files:
        
        fl_src = f"{src_dir}\\{fl}.png"
        if( os.path.isfile(fl_src)):
            fl_dest = f"{dest_dir}\\{fl}.png"
        else:
            fl_src = f"{src_dir}\\{fl}.jpg"
            fl_dest = f"{dest_dir}\\{fl}.jpg"
        shutil.copyfile(fl_src, fl_dest)
def copy_labels_to_dest(src_dir, dest_dir, src_files):
    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
    for fl in src_files:
        fl_src = f"{src_dir}\\{fl}.txt"
        fl_dest = f"{dest_dir}\\{fl}.txt"
        shutil.copyfile(fl_src, fl_dest)

def copy_images_labels_to_dest(images_dir,labels_dir, dest):
    train, val, test = split_data(images_dir,images_dir)
    train_images_dir = dest + "data/train/images/"
    train_labels_dir = dest + "data/train/labels/"
    val_images_dir = dest + "data/val/images/"
    val_labels_dir = dest + "data/val/labels/"
    test_images_dir = dest + "data/test/images/"
    test_labels_dir = dest + "data/test/labels/"
    
    copy_images_to_dest(images_dir, train_images_dir, train)
    copy_labels_to_dest(labels_dir, train_labels_dir, train)
    
    copy_images_to_dest(images_dir, val_images_dir, val)
    copy_labels_to_dest(labels_dir, val_labels_dir, val)
    
    copy_images_to_dest(images_dir, test_images_dir, test)
    copy_labels_to_dest(labels_dir, test_labels_dir, test)
    
    
images_dir = r"C:\Users\haivu\OneDrive - Sheridan College\Desktop\computer\Project\Face_Recognition\images"
labels_dir = r"C:\Users\haivu\OneDrive - Sheridan College\Desktop\computer\Project\Face_Recognition\labels"

copy_images_labels_to_dest(images_dir,labels_dir,"./")