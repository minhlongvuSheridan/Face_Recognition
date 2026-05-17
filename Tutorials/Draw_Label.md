# 1 What we have from YoLo and Computer Vision
## 4 Points of detected faces
The YOLO model returns two points to us: top-left and bottom-right. <br/>
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/69b84e4e-bf6c-4f58-9de4-ecfa1d599a21" />


From those two points, we can actually calculate the top-right and bottom-left points ourselves.<br/>
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/9532d1f5-eaa3-4310-8826-3f8020e1172b" />

## Technique to draw
All operations are based on the image in NumPy array format. You can change any pixel as you like. This means that, theoretically, you don't need to use any external functions at all and you could draw whatever you like. This is an important note to remember since we might use some custom drawing styles that are not supported by any standard library.<br/>
In this project we use ***OpenCv*** to help us draw some basic shapes. Here are some functions we will use: 
- Draw a line:<br/>
   ```python cv2.line(image, start_point, stop_point, line_color, line_thickness)``` <br/>
  Draw a short line from *start_point* to *stop_point* with specified color and thickness
- Draw a rectangle:<br/>
  ```python cv2.rectangle(image, start_point, stop_point, rec_color, rec_thickness)``` <br/>
  Draw a rectangle from top left point start_point to bottom left stop_point with specified color. If the rec_thickness > 0, then it draw the perimeter. If the rec_thickness = -1, it
  fill the whole rectangle.
- Write the text:<br/>
  ```python cv2.putText(image, text, bottom_left_start, font_type,font_scale, color, thickness)```<br/>
  Write the text with Font font_type and size font_scale. The start position is bottom left bottom_left_start
- Blend(combine the image) the image: <br/>
  ```python cv2.addWeighted(src_img1,alpha, src_img2, beta, gamma, dest_img)```<br/>
  Basically this use the formula *dest_img = src_img1 * alpha + src_img2 * beta + gamma*. The larger the coefficient, the more clearer of the image. This is the
  main technique used to create transparent rectangle. The rectangle must have alpha larger than 0.5 and the other has 1 - alpha so that it create an effect where
  the rectangle is on top of origional image. If you want to apply a mask where only relevant image is keep, background is ignore than the src_img1 must be in (0,0,0) except the
  content and beta must be 1 (1 means keep the same where the mask doesn't reach)
- Blur the image:<br/>
  ```python cv2.(src_img, kernel_size,sigmaX )```<br/>
  This is main technique to create the neon effect. Kernel is in the format (x,y) where x and y is the width and height of the matrix, respectively. Basically each pixel will be the average of thix matrix. sigMax is the standard
  deviation of axis X. It messaure the spread. Usually specified as 0 means let the function automatically apply
- Get the width and height of text size:<br/>
  ```python cv2.getTextSize(	text, fontFace, fontScale, thickness)```<br/>
  Get the height and width of the text based on the content, font type, size and thickness. It return three parameters (width,heigh), baseline. We only interested in the first tuple
  (width, height) 
  

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

Basically, we would like to display the label in an anime sci-fi style like this:
<img width="916" height="541" alt="image" src="https://github.com/user-attachments/assets/76fdf374-76b8-4c83-b2a1-ded90be95a18" />
This is just one illustration. The label position must be relative to the position of the face. For example, if the face is on the far-right side of the image, the label must be displayed on the left, not on the right like this:<br/>
<img width="810" height="719" alt="image" src="https://github.com/user-attachments/assets/e1fb5870-7b15-4287-981b-bb4377d08507" />

### Position of the label rectangle
The first thing to do is determine the position of the label rectangle. We need to determine the two points of the connector line and one corner of the label rectangle. Let's call the end of the connector attached to the face bounding box ***face_connect***, and the other end attached to the label ***label_connect***. The bottom-left corner of the label will be ***bottom_left_label***. <br/>

You might ask: why do we need the bottom-left one specifically? This is because the position of ***label_connect*** shifts dynamically, and we need a fixed reference point to place our text later. We calculate ***bottom_left_label*** directly from ***label_connect***. We want the slope of the connector line to be exactly 45 degrees.
<img width="917" height="634" alt="image" src="https://github.com/user-attachments/assets/26be5087-0cdd-4a2e-873a-576d9cdb4271" />


Our general idea is to compare the center of the face relative to the center of the image
```python
width = img.shape[0]
height = img.shape[1]
center = (width // 2, height // 2)
```
<img width="466" height="308" alt="image" src="https://github.com/user-attachments/assets/30d7f8dc-3a37-4aa4-806e-2384c497d87b" />

Also find the center of the face 
```python
center_face = ((x1+x2)//2, (y1+y2)//2)
```
<img width="392" height="346" alt="image" src="https://github.com/user-attachments/assets/e3302763-4fab-46dc-860a-181208e2087e" />

#### Case 1: Face is on the top right
<img width="732" height="598" alt="image" src="https://github.com/user-attachments/assets/ad6556b4-65fb-4b2d-950a-dc9c61a79c4c" />

```python
******face_connect****** = (x1,y2)
label_connect = (******face_connect******[0] - delta_x, ******face_connect******[1] + delta_y)
bottom_left_label = (label_connect[0] - rect_width, label_connect[1] + rect_height)
```

#### Case 2: Face is on the top left
<img width="673" height="587" alt="image" src="https://github.com/user-attachments/assets/643f5a6f-9476-47b5-8896-027d886f9bec" />

```python
******face_connect****** = (x2, y2)
label_connect = (******face_connect******[0] + delta_x, ******face_connect******[1] + delta_y)
bottom_left_label = (label_connect[0], label_connect[1] + rect_height)
```
#### Case 3: Face is on the bottom right 
<img width="716" height="615" alt="image" src="https://github.com/user-attachments/assets/4d5cc5c4-54da-4893-b482-72e234d0f4be" /><br/>

```python
******face_connect****** = (x1,y1)
label_connect = (******face_connect******[0] - delta_x, ******face_connect******[1] - delta_y)
bottom_left_label = (label_connect[0] - rect_width,label_connect[1])
```
### Case 4: Face is on the bottom left
<img width="678" height="616" alt="image" src="https://github.com/user-attachments/assets/a47c51af-ea3b-481f-b168-3404558dd177" /> <br/>

```python
******face_connect****** = (x2,y1)
label_connect = (******face_connect******[0] + delta_x, ******face_connect******[1] - delta_y)
bottom_left_label = label_connect
```


### Draw the text
Before drawing the text, we need to position it inside the label rectangle. <br/>
Calculate the height and the width of text<br/>
<img width="406" height="163" alt="image" src="https://github.com/user-attachments/assets/3719b279-fd74-4576-9e13-98b62620bd93" /><br/>

```python
((text_width, text_height), _ = cv2.getTextSize(Text,text_font, text_size, text_thickness))
```
Assume that the user already provided the spacing and padding of the text inside the box. Recall the bottom_left_label that we calculate
in the previous section. Now we calculate the position of text. I will work on the case of three text, i would be similiar for any number of text
<img width="546" height="507" alt="image" src="https://github.com/user-attachments/assets/dc1ece5b-5b60-46c8-870d-72714bd631bd" />


```python
text_1_start_point = (bottom_left_label[0] + padding, bottom_left_label[1] - padding)
text_2_start_point = (text_1_start_point[0], text_1_start_point[1] - text_height - spacing)
```



We now can also calculate the width,height, top left, and bottom right of the label
```python
max_width = max(text_width_1, text_2_width)
max_height = text_height
rec_width = text_width + 2 * padding
rec_height = 2 * text_height + 2 * padding + line_spacing
label_top_left = (bottom_left[0], bottom_left[1] - rec_height)
label_bottom_right = (bottom_left[0] + rec_width, bottom_left[1])
```



Now we are pretty much done with preparing all the thing. But it one thing is missing. That is the background of the text inside the label to differentiate it from the background
of the image. We can solve it by fill in some 
### Blending images
To make the background for image text, we can just set the **thickness* parameter of **cv2.rectangle** to -1. But doing that will overide the origional image. 
<img width="975" height="265" alt="image" src="https://github.com/user-attachments/assets/7e709383-3736-4007-a447-2483ad9680cf" /> <br/>
While it is okay, but we want more than that. We want it to be transparent label so that it resemble the glass-style monitor
<img width="991" height="264" alt="image" src="https://github.com/user-attachments/assets/bfc624ae-7270-41fb-9607-80d37b679319" />
Remember that if though it appear that the label image is on top of origional image, there is actually only single pixel that contains RGB channel. nothing on stop of anything. To create the illusion, we use a technique call "blending images" or basically just combine them. Specially we will use linear blending <br/>
$$g(x) = (1 - \alpha)f_1(x) + \alpha f_2(x)$$ <br/>
where $$f_1$$, $$f_2$$, and $$g(x)$$ are the images color of the **src_img1**, **src_img2**, and **final_img**,respectively. The $$\alpha$$ is the transparency. <br/>
If we want the **src_img2** to appear on top of **src_img1**, we need to set $$\alpha$$ to be larger than 0.5<br/>
- Step 1: Create another image layer with the same size at the label rectangle<br/>
  ```python overlay = image.copy()```
- Step 2: In the new layer, fill in with you desired color. Remember to set thickness to -1 <br/>
  ```python cv2.rectangle(image, label_top_left,  label_bottom_right, border_color, -1)```
- Step 3: Combine both the layers, let the final result overide the origional image.<br/>
  ```python cv2.addWeighted(overlay,0.6,image,1- 0.6,0,image) ```

We now can write the text and draw the rectangle perimeter
```python
cv2.putText(image,Text_1, text_1_start_point ,style_text, text_size ,text_color, text_thickness)
cv2.putText(image,Text_2, text_2_start_point ,style_text, text_size ,text_color, text_thickness)
cv2.rectangle(image, label_top_left,  label_bottom_right, border_color, border_thickness)
```
# 3 Gaussian Blur: Drawing with neon effect
We are now done with label things. But it looks somewhat boring now. We want more than that. We want the text look like as if it was from the actual monitor light <br/>
<img width="176" height="90" alt="image" src="https://github.com/user-attachments/assets/b5652892-e419-4df3-8684-f5e7ae5a3586" /> <br/>
The technique used to create such that Neon Effect is mainly Gaussian Blur. The general idea is to blur the thicker image to create a fealing of spreading light center around a thinner light source text. The thicker part is called outer glow where it has darker color to represent distance from light source. The centered text is called inner glow that has lighter color to represent the light source. In some situation, we can even let the inner glow to be white to represent really strong light source. <br/>
- Step 1 Generally we create new dark mask. It is dark so the area that is not related to the neon effect won't affect the origional image. (0,0,0) cancel out its coefficient  <br/>
  ```python
  mask_neon = image.copy()
  mask_neon[:] = (0,0,0)
  ```
- Step 2 Run a for loop to draw the blurred texts. The idea is to draw the thickest first until thinnest text. Number of iterations depend on the strength of neon effect you want. 
  ```python
  for i in range(text_thick + text_strength, 1, -1):
    cv2.putText(mask_neon,Text, text_start,style_text, text_size ,outer_text_color, i)
    mask_neon = cv2.GaussianBlur(mask_neon,(2*i +1, 2*i + 1), 0)
  ```
  (2*i +1, 2*i + 1) is the matrix surrounding the pixel that it will look at and take the average value of them. The bigger the more blurred it will be. The 0 means let the function calculate automatically based on kernel size
- Step 3: Blend the mask back to the origional image
  ```python
  cv2.addWeighted(mask_neon,0.7, image, 1,0,image)
  ```
  It is 1 which means keep the same image. So we don't do the transparent any more(try 0.3 you will see that it is behind dark layer). 0.7 of mask_neon will make it feel like there is a light right there. Don't worry if the background of origional image is in front because our mask is based on the transparent background label. We basically make this background label lighter at some point but it is still above the origional image
   Step 4: Write the lighter color text to represent the light source
  ```python
  cv2.putText(image,text, text_start,style_text, text_size , inner_text_color, text_thick)
  ```
We are done with all technical detail. The actual code in draw_box.py, especially the ***draw_neon_label***, is kinda messed up due to the optimization. However, all the concepts are explained so it should be easy to spot it out. Thank you
