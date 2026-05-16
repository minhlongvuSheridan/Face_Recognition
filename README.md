# Run scripts
There are two files: **main.py** for face recognition with tracking and **facenet_no_sort.py** for face recognition without tracking.
To run scripts, open the Face_Recognition directory and type: <br/>
```python ./main.py```
or <br/>
```python ./facenet_no_sort.py```
# Demo

<table>
  <tr>
    <td>
      <p align="center"><b>With SORT Tracking</b></p>
      <video src="https://github.com/user-attachments/assets/45ab2e4d-7f22-49aa-ad88-656c24732aac" width="60%" controls></video>
    </td>
    <td>
      <p align="center"><b>Without SORT Tracking</b></p>
     <video src="https://github.com/user-attachments/assets/6237a0d0-25b3-4348-b2c4-919bcae4130e" width="60%" controls></video>
    </td>
  </tr>
</table>
With basic tracking, the FPS is improved from below 5 all the time to around 25 most the time

# Dependencies


# Customize the identity 
If you want to add new identity, create a new folder with the target's name in the identity folder. 
<img width="1908" height="378" alt="image" src="https://github.com/user-attachments/assets/fb8f831f-eaf0-4b55-93a0-0a9d4cc8c3f7" />
I found out that around 20 images of an identity should be good for face recognition. Remember to crop the image to take only the face
so the model can only learn from the face not the surrounding enviroment
<img width="1461" height="521" alt="image" src="https://github.com/user-attachments/assets/bb198405-a257-4713-a469-69f366ce5616" />

# Trainning YoLo model 
The YOLO model was trained on 300 images on the yolov8l.yaml. It is fine but it could be better with more data. I suggest adding more 700 images of different faces, even better if they are augmented. <br/>
If you intend to train better YOLO model, generally there are four steps:
- Step 1: Delete the *data* folder 
- Step 2: Add images to *images* and labels to *labels* folders
- Step 3: Run the script *split_data.py* to split the data
- Step 4: Run  ```yolo task=detect mode=train model=yolov8l.yaml data=/content/drive/MyDrive/Data/FaceRecognition/data/config.yaml epochs=100``` <br/>
I have written a detailed step-by-step guidance of how to do this in *Tutorials*, feel free to read it

