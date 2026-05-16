# 1 What we have from YoLo and Computer Vision
## 4 Points
The yolo return us two points. Top left and bottom right
<img width="691" height="578" alt="image" src="https://github.com/user-attachments/assets/69b84e4e-bf6c-4f58-9de4-ecfa1d599a21" />


From those two points, we can actually calculate the top right and bottom left ourself
<img width="817" height="607" alt="image" src="https://github.com/user-attachments/assets/9532d1f5-eaa3-4310-8826-3f8020e1172b" />

## Technique to draw
All the operations are based on the image in the numpy array format. You can change any pixel as you like. This mean that, theoretically, you don't need to use any external function at
all and you could draw whatever you like. This is important note to remember since we might use some drawing that is not supported by any lirary<br/>
In this project we use ***OpenCv*** to help us draw some basic shapes. Here are some functions we will use: 
- Draw a line: ```python cv2.line(image, start_point, stop_point, line_color, line_thickness)``` <br/>
  Draw a short line from *start_point* to *stop_point* with specified color and thickness
- Draw a rectangle: ```python cv2.rectangle(image, start_point, stop_point, rec_color, rec_thickness)``` <br/>
  Draw a rectangle from top left point start_point to bottom left stop_point with specified color. If the rec_thickness > 0, then it draw the perimeter. If the rec_thickness = -1, it
  fill the whole rectangle.
- Write the text: ```python cv2.putText(image, text, bottom_left_start, font_type,font_scale, color, thickness)```
  Write the text with Font font_type and size font_scale. The start position is bottom left bottom_left_start
- Blend(combine the image) the image: ```python cv2.addWeighted(src_img1,alpha, src_img2, beta, gamma, dest_img)```
  Basically this use the formula *dest_img = src_img1 * alpha + src_img2 * beta + gamma*. The larger the coefficient, the more clearer of the image. This is the
  main technique used to create transparent rectangle. The rectangle must have alpha larger than 0.5 and the other has 1 - alpha so that it create an effect where
  the rectangle is on top of origional image. If you want to apply a mask where only relevant image is keep, background is ignore than the src_img1 must be in (0,0,0) except the
  content and beta must be 1 (1 means keep the same where the mask doesn't reach)
- Blur the image: ```python cv2.(src_img, kernel_size,sigmaX )```
  This is main technique to create the neon effect. Kernel is in the format (x,y) where x and y is the width and height of the matrix, respectively. Basically each pixel will be the average of thix matrix. sigMax is the standard
  deviation of axis X. It messaure the spread. Usually specified as 0 means let the function automatically apply
  

That is pretty much everything we need from opencv
# 2 Drawing without neon
Before going to these fancy neon effect. We need to understand why we have to draw something first
## Draw the 4 corners
Our idea is not to draw a full side of rectangle but just a short line originated from every corners 
<img width="464" height="390" alt="image" src="https://github.com/user-attachments/assets/7c6c2a2e-db46-47d7-80a6-d9b0cfbfab65" /> <br/>
We already calculated the position of each corners. To draw the short line, we need to calculate the end point of each short line. To do that, we need to know what is the length of 
short line that we expect. I suggest to take the portion of the full side. For instance, we could take one sixth. It would be weird if the all short lines are at different length. Thus
I will choose the smalles between delta_x and delta_y
```python
delta_x = x2 - x1
delta_y = y2 - y1
l = delta_x if delta_x <= delta_y else delta_y
```
For each corner, we will have to calculate stop point based on the length
```python
# top left
top_left_vertical = (x1, y1 + l)
top_left_horiozntal = (x1 + l, y1)

# Top Right
top_right_vertical = (x2, y1 + l)
top_right_horizontal = (x2 - l, y1)

# Bottom left
bottom_left_vertical = (x1, y2 - l)
bottom_left_horizontal = (x1 + l, y2)

# Bottom right
bottom_right_vertical = (x2, y2 - l)
bottom_right_horizontal = (x2 - l, y2)
```
<img width="729" height="604" alt="image" src="https://github.com/user-attachments/assets/e7f7f5be-2de3-4d3e-932c-7860b9cd84f0" />

When we have all the necessary points. Just use the ```python cv2.line()``` to draw all the lines. You might need to call it 8 times
## Draw the label

### Blending images
### 

# 3 Drawing with neon effect
Even though the code in the file is the mess up (it does have resonable rationale of optimizing). There is a clear pattern of doing the image.
