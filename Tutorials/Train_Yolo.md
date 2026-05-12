# 1 Collect Images
The current dataset now has 300 images. However it is consider small dataset. It would be better if have around thoudsands images so it could generalize for many situations. The choice of images is up to you but do remember that the images should be in many angles and different enviroment. <br/>
<img width="628" height="448" alt="image" src="https://github.com/user-attachments/assets/6533e751-8dfe-4afd-a216-25ea0eb63011" /><br/>
Figure 1. Good and clear image <br/>
This is an clear image about Elon Musk which is good for our face detection and recognition. However, when we test yolo model later, if you use Elon Musk image in front of camera, the YOLO couldn't recognize it because it was trained on good quality image not bad one. Thus, we need to also train it on the camera perspective as well like this. <br/>
<img width="1278" height="729" alt="image" src="https://github.com/user-attachments/assets/5a02392b-eea0-4d6e-8118-06be9edeab0e" />
Figure 2. Image taken from the laptop camera<br/>
<img width="483" height="629" alt="image" src="https://github.com/user-attachments/assets/18b88e54-d365-4ff8-9e12-b1e104f22205" /><br/>
Figure 3. Image taken from the phone camera<br/>
Thus, I suggest for each good quality image, you should take two more from laptop camera and from the phone camera.
# 2 Annotate the images using cvat
After collecting desired number of images. Let first put all of them into single folder call "images". The problem now is that we have images but we don't have any way to tell the detection model what is a face. Thus, we need a way to name or annoate images. <br/>
There are many tools help us to do that: Roboflow, CVAT, LabelImg. For me, I used CVAT since it worked for me.

# 3 Organize the images
### 3.1 Split using Scikir-learn
### 3.2 Structure to YOLO format 

# 3. Train the Model on Google Colab


