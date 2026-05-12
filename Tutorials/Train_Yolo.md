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
For the sake of simplicity, we only do 3 images from the section 1. After collecting desired number of images. Let first put all of them into single folder call "images".
<img width="1140" height="382" alt="image" src="https://github.com/user-attachments/assets/6cf7ec55-bae9-487b-aa8b-2767eb85c3f2" /> <br/>


The problem now is that we have images but we don't have any way to tell the detection model what is a face. Thus, we need a way to name or annoate images. <br/>
There are many tools help us to do that: Roboflow, CVAT, LabelImg. For me, I used CVAT since it worked for me. <br/>
- Step 1 Go to the web https://www.cvat.ai/ and sign in 
- Step 2 Go to "Task" and click on the "+" button on the right. Then click on the "Create on new task"
<img width="1154" height="444" alt="image" src="https://github.com/user-attachments/assets/b63f49c4-92ef-4b3f-9ad5-6d2dec003677" />

- Step 3 Name the project as you want. Since we only need face class so I only added "Face" label. Then upload all the images in the "images" folder in the step 1. then click on "Submit & Open"
<img width="817" height="932" alt="image" src="https://github.com/user-attachments/assets/c240c4ab-a8c9-40e7-a2e8-69a6bbac60f0" /> <br/>
  Depend on how many images you have, you could need to wait few seconds for few images or minutes for hunred of images. Just until the browser navigate to new pages
  
- Step 4 On the new page, Click on the "Job #..."
<img width="1790" height="755" alt="image" src="https://github.com/user-attachments/assets/686a9499-2839-4957-bb2f-211f9b52c4e4" />

- Step 5 Click on "Draw a Rectangle" and then click on "shape"
<img width="1870" height="978" alt="image" src="https://github.com/user-attachments/assets/2d08d737-601f-44db-92d2-ed983177ffd8" />

- Step 6 Since we are drawing a rectangle, we only need two points where one define the bottom right corner and the other define top left corner. Thus, we need to left click on the image to define a point.
<img width="1910" height="1008" alt="image" src="https://github.com/user-attachments/assets/1c2e7f5f-1c78-46ab-8fc4-1bbf3d66e3a2" />
<img width="1913" height="1017" alt="image" src="https://github.com/user-attachments/assets/9e28473f-85c6-407b-a838-a8d3cba8e5f0" />
<img width="1919" height="997" alt="image" src="https://github.com/user-attachments/assets/c4ee250d-e972-44fd-a456-a027fceeceab" />

- Step 6 Press "Ctrl + s" to save your work
- Step 7 Click on the "Task" to exist
<img width="1128" height="86" alt="image" src="https://github.com/user-attachments/assets/9e25f384-0d80-4914-aaec-22b76d04539d" />

- Step 8 To the left of the actions, Click on the three dots
<img width="1134" height="135" alt="image" src="https://github.com/user-attachments/assets/303e821c-12fd-4c30-b73f-e29d4d251868" />

- Step 9 on the dropdown, click on "Export task dataset
<img width="335" height="424" alt="image" src="https://github.com/user-attachments/assets/4c9c50c6-5815-4244-9fcc-a5543534a706" />

- Step 10 On the pop up window, choose "YOLO 1.1" format. The custom name is optional but you should have one because you could have weird long code. Then click on OK
<img width="528" height="368" alt="image" src="https://github.com/user-attachments/assets/bbbb9400-b3f8-48a1-bff2-4945a38d4120" />

- Step 11 Go to the "Request", check the time stamp to correctly identify your request. click on the vertical three dots and click download
<img width="1901" height="242" alt="image" src="https://github.com/user-attachments/assets/f4691b85-0ee2-4873-a964-fd02a6e0faba" />

- Step 12 Extract the "face.zip" file and open the face folder. Open the "obj_train_data" and copy all the files in there
<img width="1305" height="395" alt="image" src="https://github.com/user-attachments/assets/1226c46a-2001-4db7-8419-da835c9d3c0e" />

- Step 13 Open the same folder where it store "images" in the section "Collect Images" section. Create a folder called "labels" and paste all the images from the step 12 to it
<img width="1157" height="413" alt="image" src="https://github.com/user-attachments/assets/4ec78e28-e3b7-40f0-834b-6f61fd0b09bb" />
<img width="1143" height="419" alt="image" src="https://github.com/user-attachments/assets/2f775f75-f548-4de5-b8e8-06b6c496d6cc" /> <br/>

what we copied and paste in step 12 and 13 are labels of the images. When we click on a image we will see a row of 5 numbers like this
<img width="507" height="144" alt="image" src="https://github.com/user-attachments/assets/10ca9d60-d8cc-4120-b634-d69782dd22fe" /><br/>

FIrst of all, each row is the number of labeled object in an image. In our case, we only label a face for each image so we only have a row. If we label multiple faces in an image then we should expect more. <br/>
Secondly, the number "0" represent for the class name where 0 means the first class. If we have more classes such as leg, arm or body then they would be 1,2, or 3, respectively <br/>
Thirdly, the two next numbers represent the center point (x,y) of the rectangle of that detected object.  <br/>
Finally, the last two numbers represent for the width and height of the rectangle <br/>




# 3 Organize the images
### 3.1 Split using Scikir-learn
We now have *images* and *labels* for our model. However, it is not ready for training process yet. As far as I see, we need to split them into train, validation and testing subsets.
- **Training dataset**: the main purpose to build the model. It is used to change the parameters of the model. Think it is as "lecture slides"
- **Validation dataset**: This is used as to check overfitting of the training dataset. After every epoch or iteration of training, it is used to calculate metric loss. Think it is as "practice exam:
- **Testing dataset**: this is used as final evalutaion of the model. It is used only when the model finish all the epochs. Think it is as "final exam"
There are many ways of splitting those sets. For this tutorial, I choose the 70-15-15 splitting which means 70% for the training, 15% for the valdiation, and 15% for the testing, respectively.<br/>

To ease the process of splitting dataset, I use the ```train_test_split``` function from the sklearn. Notice that the images and labels they have similiar names and only differ in extension. Our algorithm is to take all the file names and truncate away the extension. Then split it by using the ```train_test_split```. Since we have three subsets, we need to split 2 times
#### 3.1.1 First split
   We will split the origional dataset into 85 for train_val and 15 for testing.
```{python}
train_val, test, _, _ = train_test_split(
    names, names, 
    test_size = 0.15, 
    random_state=42
    )
```
The ```train_test_split``` usually take features and labels but we only need the splitting features so it is basically same thing for us. ```_``` means we discard the results
#### 3.1.2 Second split
In the second split, we will split *train_val* into *train* and *val*. The first split is quite obvious where we can specify directly how much we want for the test_size. However, the second split is kinda tricky because *train_val* is just equal to *85* of the orgiional dataset so if we specify 15% for the *test_size*, it just result in 0.125 * 85% = 12,75% of the origional dataset for the *val*. This result is not expected since we want the *val* to be 15%. <br/>
**Solution**<br/>
Assume that N is the total rows where *train*,*val*, and *test* are the subsets where *train* + *val* + *test* = N<br/>
We are doing 70-15-15 so we expect *train* = 0.7N, *val* = 0.15N and *test* = 0.15N (1) <br/>
After the first split we have *train* + *val* = 0.85 N <br/>
Lets call x be value of *test_size* of second split. This x basically the cut portion of the *train_vaL* for the *val*<br/>
*val* = x * *train_val* = x * 0.85N (2) <br/>
From (1) and (2) we have <br/>
x * 0.85N = 0.15N => x = 0.15 / 0.85 ~~ 17.65%<br/><br/>

As shown above, for the second split, we need to specify *0.1765* instead of "0.15"

```{python}
train, val, _, _ = train_test_split(
    train_val, train_val, 
    test_size = 0.1765, 
    random_state=42
    )
```
### 3.2 Structure to YOLO format
For now we have all the subsets. But they are just the names which contains no useful information. We will base on this and move the corresponding files to the structure below
<img width="367" height="625" alt="image" src="https://github.com/user-attachments/assets/6c223c9b-6382-48a9-aae5-aa4abed24dc9" />
We can do this either manually or writing a script base on the name. If you write a script, be remember the images could be jpg, png or jpeg extensions.
### 3.3 config.yaml
This is the file that is specified when we train the model. Its job is to tell where the data reside and what are the class names. It might be little confused where we already put it in the "data" directory so why we need to specify it a gain? Basically, the config.yaml file could be anywhere. We could even put it in the same directory with the "data" folder and still work. We put it in inside the "data" is just general convention
```{yaml}
path: C:\Users\haivu\Downloads\train_yolo\data
train: train/images
val: val/images
test: test/images
nc: 1
names:
  0: Face
```
- the *path* is the path to the *data* directory. 
- The *train*, *val*, and *test* are relative path inside that *data* to their coressponding path.
- nc is the number of classes. We only have one class so it is 1
- names: is the class name. 0 means the first class. Since we only have one class so 0 correspond to Face class

# 3. Train the Model on Google Colab
It is the most convinient if we can just train the model locally because we could just train as many epochs as we want. However, I have some problems with my GPU so I will use the Google Colab to train for it<br/>
- Step 1: Upload the whole *data* folder to the google drive
  
- Step 2: Open Google Colab using the same email that we upload the google drive
  
- Step 3: Open New NoteBook
  
- Step 4: Click on "Files" and then click on "Mount Google Drive"
<img width="1869" height="745" alt="image" src="https://github.com/user-attachments/assets/5be5f640-1465-4fde-877c-c0b80544782c" />

- Step 5: On the pop up window, Choose "Connect to Google Drive"
  
- Step 6: Wait a bit until the icon "drive" pop up. This is our goolge drive storage
<img width="359" height="286" alt="image" src="https://github.com/user-attachments/assets/134e7875-8e1a-40ac-a83c-b7b0d465df98" />

- Step 7: Find the *config.yaml* and open it. Copy the path of the *data* directory and paste it to the *path* variable of the left panel. Then "Ctrl +S" to save
<img width="1918" height="967" alt="image" src="https://github.com/user-attachments/assets/ffdb5880-3fdd-44ea-b6d7-c8c232d435be" />


- Step 8: Go to *Runtime* and click on *Change runtime type"
<img width="750" height="730" alt="image" src="https://github.com/user-attachments/assets/a95a97d6-1c80-4d6b-8530-ab3913b5b4da" />

- Step 9: On pop up window, choose T4 GPU and click "Save"
<img width="672" height="657" alt="image" src="https://github.com/user-attachments/assets/ff63c264-0cb2-4b90-bbe9-25b7fe49f00a" />

- Step 10: Open command cell and install YOLo library ```!pip install ultralytics```. The *!* means it run on terminal
  
- Step 11: Open new command cell, type ```!yolo task=detect mode=train model=yolov8l.yaml data=/content/drive/MyDrive/Data/FaceRecognition/data/config.yaml epochs=60``` to run the training process. the data arugments is the path to the *config.yaml file* not the *data*. The model *yolov8l.yaml*, *v8* means version 8. l in *8l* means large weiight. *.yaml* means we train it from scartch. We could train from pre-trained model by using *yolov8l.pt*. The epochs is how many times we want to train. For my experience, the epochs should be somewhere above 50. It should not be two big since there is limited time on google colab.
  
- Step 12: Wait until we see the result like this
<img width="1386" height="278" alt="image" src="https://github.com/user-attachments/assets/ee605642-7f20-4643-8dca-97682f1a0233" />

- Step 13: After training, there exists "runs" folder. this folder store all the results of each training process. Each time we run command it will named like *train*,*train-1*,... where larger number means more recent. Open the *train-..* that can just run. Our model is store in *weight*. The last.pt means the model in last epoch whereas best.pt means best performance. we would like to choose *best.pt*
<img width="345" height="144" alt="image" src="https://github.com/user-attachments/assets/6da9c0c8-2058-4aaf-9fc0-b5d586a58154" />

<img width="324" height="397" alt="image" src="https://github.com/user-attachments/assets/47b2abc5-4b2d-40a8-8055-f5c9d61af8d4" />

- Step 14: Downlaod the *best.pt*, rename it to *yolo_detection.pt* and store in our local folder
<img width="948" height="243" alt="image" src="https://github.com/user-attachments/assets/aad982f7-b4c6-43fe-8cda-550766c2f861" />

- Step 15: Create script files<br/>
Create a file called *draw_box.py* with the code
  ```python
  import cv2
  def draw_corners(image, x1, y1, x2, y2, length, thick):
      l = length
      t = thick
      green_retro = (44,255,5)
      
      # Top Left
      cv2.line(image, (x1, y1), (x1 + l,y1), green_retro, t )
      cv2.line(image, (x1, y1), (x1,y1 + l), green_retro, t )
      
      # Top Right
      cv2.line(image, (x2, y1), (x2 - l, y1), green_retro, t )
      cv2.line(image, (x2, y1), (x2, y1 + l), green_retro, t )
      
      # Bottom Left
      cv2.line(image, (x1, y2), (x1 + l, y2), green_retro, t )
      cv2.line(image, (x1, y2), (x1, y2 - l), green_retro, t )
      
      # Bottom Right
      cv2.line(image, (x2, y2), (x2 - l,y2), green_retro, t )
      cv2.line(image, (x2, y2), (x2,y2 - l), green_retro, t )
  
  
  def draw_label(image, x1, y1,label, color_text, color_box):
      overlay = image.copy()
      (text_width, text_height), _ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
      
      cv2.rectangle(overlay,(x1 - 5,y1 - 15 + 10 ),(x1 + text_width + 5,y1 - text_height - 15 - 10), color_box,-1)
      cv2.addWeighted(overlay,0.6,image,1- 0.6,0,image) 
      cv2.putText(image,label, (x1, y1 - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.8,  color_text, 1)
  ```
  And then create a file name ***main.py***
  ```python
  import cv2
  from ultralytics import YOLO
  from draw_box import *

  lap_camera = cv2.VideoCapture(0)
  model = YOLO("./detection.pt")

  while True:
      success, img = lap_camera.read()
      if success == False:
          continue
      results = model(img, stream = True)
      for result in results:
          for box in result.boxes:
              x1, y1, x2, y2 = map(int, box.xyxy[0])
              
              side_length = x2 - x1
              length = side_length // 6
              
              conf = int(float(box.conf[0]) * 100)
              draw_corners(img, x1,y1,x2,y2,length,2)
              draw_label(img, x1, y1, f"Face: {conf:.2f}",(65,255,0),(0,0,0)) 
              
      cv2.imshow("Image", img)
      cv2.waitKey(1)
  ```
- Step 16: Run the *main.py* and test the model. Repeat all the steps since section 1 until the model work as your desire








