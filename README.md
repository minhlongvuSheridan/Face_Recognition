<h1 align="center">🕵️‍♂️Face Recognition with YOLO, SORT, and FaceNet+SVM🕵️‍♂️</h1>

# Description📝
This is the project that mark my first step in AI field. If you look at several videos on *Face Recognition* on youtube, you will see that their speed is kind of slow. This is because the face recognition stage is inherently slow.  <br/>
<p align="center">
  <img width="1380" height="238" alt="image" src="https://github.com/user-attachments/assets/5882728a-6a1b-4746-aea6-b9a086202997" />
  <br>
  <b>Figure 1. YOLO + (FaceNet + SVM)</b>
</p>

The question "Where is the face" is really cheap since YOLO could detect multiple faces with the same cost of around 30ms. However, the question "what does the question look like" is quite expensive since FaceNet has to run its complex neural network to extract the feature. Each face could take around 60ms. For every new frame, the usual tutorials do the recognition right after the recognition.  <br/>
<div align="center">
  <img width="1449" height="316" alt="image" src="https://github.com/user-attachments/assets/23d6477a-59ca-48dd-a763-014bcbd5aac3" />

  <p>Figure 2. YOLO + SORT + (FaceNet + SVM)</p>
</div>
As an solution for that problem, I add the tracking phase where we will ask "Is this the same face" before going to recognition. This question of tracking is just as cheap as detection. If this the same face, it will bypass the recognition and display it directly
If it is not, the model will perform recognition. If the model is confident in the result (highest probability and > 60), the face will be tracked and won't go through the recognition again as long as it keep its ID. The result is really good with the improvement from 5 FPS up to 25 FPS almost all the time.<br/> 

# Run scripts🚀
There are two files: **main.py** for face recognition with tracking and **facenet_no_sort.py** for face recognition without tracking.
To run scripts, open your terminal and run those commands
```python
# 1. Clone Face_Recognition repo
git clone https://github.com/minhlongvuSheridan/Face_Recognition.git
cd Face_Recognition

# 2. Install all dependencies from requirements.txt
pip install -r requirements.txt

# 3. Run the Face_Recognittion app
python main.py
# If want to run no tracking: python facenet_no_sort.py
```
# Demo📊

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

# GPU notes
If you only run it on CPU, it might be extremely slow. Both my demo run on the GPU. Ensure that you already installed **CUDA Toolkit** and **cuDNN**. <br/>
If you don't know how, please watch at *1:09:02* to this tutorial **https://www.youtube.com/watch?v=WgPbbWmnXJ8&t=11529s**. He will guide you how to do it

# Customize the identity👤
If you want to add new identity, create a new folder with the target's name in the identity folder. 
<img width="1908" height="378" alt="image" src="https://github.com/user-attachments/assets/fb8f831f-eaf0-4b55-93a0-0a9d4cc8c3f7" />
I found out that around 20 images of an identity should be good for face recognition. Remember to crop the image to take only the face
so the model can only learn from the face not the surrounding enviroment
<img width="1461" height="521" alt="image" src="https://github.com/user-attachments/assets/bb198405-a257-4713-a469-69f366ce5616" />

# Trainning YoLo model🏋️‍♂️ 
The YOLO model was trained on 300 images on the yolov8l.yaml. It is fine but it could be better with more data. I suggest adding more 700 images of different faces, even better if they are augmented. <br/>
If you intend to train better YOLO model, generally there are four steps:
- Step 1: Delete the *data* folder 
- Step 2: Add images to *images* and labels to *labels* folders
- Step 3: Run the script *split_data.py* to split the data
- Step 4: Run  ```yolo task=detect mode=train model=yolov8l.yaml data=/content/drive/MyDrive/Data/FaceRecognition/data/config.yaml epochs=100``` <br/>
I have written a detailed step-by-step guidance of how to do this in *Tutorials*, feel free to read it

