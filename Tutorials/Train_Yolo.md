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
### 3.2 Structure to YOLO format 

# 3. Train the Model on Google Colab


