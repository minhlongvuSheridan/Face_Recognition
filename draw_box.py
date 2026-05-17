'''
Author: Minh Long Vu
Application: This functions draw the customized bounding box instead of traditional rectangle one
'''

import time
import math
import cv2

# ------------------------------- Neon effect ------------------------------------
def find_center_image(img):
    
    """
    find_center_image: calculate of the center of an images
    img: the numpy array of image
    return: Position tuple of center point
    """
    
    
    width = img.shape[0]
    height = img.shape[1]
    x_center = width // 2
    y_center = height // 2
    return (x_center, y_center)

def get_points_for_connector(img,x1,y1,x2,y2, length, rect_width, rect_height):
    
    
    """
    get_points_for_connector: get the two ends of connectors and bottom left of the label
    img: numpy array of the orgional image
    (x1,y1): Top left position of the detected face
    (x2,y2): Bottom right position of the detected face
    length: the length of the connector line
    rect_width: the width of the label rectangle 
    rect_height: the length of the label rectangle
    return: Return one connected to detected object, the other end connected to label 
    and bottom left of label
    """
    
    
    
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    center = (x_center, y_center) # center of detected object
    delta_x = int(length * math.cos(math.pi /4 ))
    delta_y = int(length * math.sin(math.pi /4 ))
    
    
    img_center = find_center_image(img) # center of origional image
    
    """
    face_connect:  An end of connector connected to detected object
    label_connect: An end of connector connected to label rectangle
    bottom_left_label: bottom left of label rectangle
    """    
    if center[0] > img_center[0]  and center[1] > img_center[1]:
        face_connect = (x1,y1)
        # bottom right
        label_connect = (face_connect[0] - delta_x, face_connect[1] - delta_y)
        # bottom left
        bottom_left_label = (label_connect[0] - rect_width,label_connect[1])
    elif center[0] > img_center[0] and  center[1] < img_center[1]:
        face_connect = (x1,y2)
        # top right
        label_connect = (face_connect[0] - delta_x, face_connect[1] + delta_y)
        # bottom left
        bottom_left_label = (label_connect[0] - rect_width, label_connect[1] + rect_height)
        
    elif center[0] < img_center[0]  and center[1] > img_center[1] : 
        face_connect = (x2,y1)
        # bottom left
        label_connect = (face_connect[0] + delta_x, face_connect[1] - delta_y)
        # bottom left
        bottom_left_label = label_connect
    else:
        face_connect = (x2, y2)
        # tio left
        label_connect = (face_connect[0] + delta_x, face_connect[1] + delta_y)
        # bottom left
        bottom_left_label = (label_connect[0], label_connect[1] + rect_height)
    return face_connect, label_connect, bottom_left_label

def get_width_height_text(Name, Job, Country, text_size):
    
    """
    get_width_height_text: This application get the maximum height and width among three texts: 
    Name, Job, and Country to calculate the length and height of label rectangle
    Name, Job, Country: Texts that want to display on 
    text_size: the size of the text
    return: max witdh and max height
    """
    
    
    (Name_width, Name_Height), _ = cv2.getTextSize(Name,cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)
    (Job_width, _), _ = cv2.getTextSize(Job,cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)
    (Country_width, _), _ = cv2.getTextSize(Country,cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)
    max_height = Name_Height
    max_width = Name_width
    if max_width < Job_width:
        max_width = Job_width
    if max_width < Country_width:
        max_width = Country_width
    print(f"name: {Name_width} job: {Job_width} country {Country_width}")
    return max_width, max_height

def get_points_for_text(rec_bottom_left, text_height, spacing, padding):
    
    """
    get_points_for_text: Get the 3 bottom left points needed to put text for three 3 texts
    rec_bottom_left: bottom left of rectangle label
    text_height: the height of the text
    spacing: the space between two lines
    padding: the space between the line the the rectangle
    return: three points for three texts top to down: name, job, and country
    """
    
    
    country_point = (rec_bottom_left[0] + padding,
                     rec_bottom_left[1] - padding)
    job_point = (country_point[0],
                 country_point[1] - text_height - spacing)
    name_point = (job_point[0],
                  job_point[1] - text_height - spacing)
    return name_point, job_point, country_point


def draw_neon_corners(image, x1, y1, x2, y2, inner_color,outer_color, neon_strength,length, thick):
    
    
    """
    draw_neon_corners: For any corner of four conects, draw a vertical and horizontal lines instead of full side of
    rectangle 
    (x1, y1): Top left of the detected object bounding box
    (x2, y2): Bottom right of the detected object bounding box
    inner_color: the center non-blured line which act as light source
    outer_color: the outer blurred lien which act as spreading light
    neon_strength: the strength of the spreading light effect, larger will spreading further
    length: the length of the short line
    thick: the thickness of the line
    return: no return
    """
    
    
    
    
    start = time.time()
    # rename for shorter word
    l = length
    t = thick
    green_retro = (44,255,5)
    mask_neon = image.copy()
    mask_neon[:] = (0,0,0) # didn't affect anything if blend
    # Step 2: Apply blurred text, the bigger the more blurred
    
    for i in range(thick + 2* neon_strength, thick, -2):
        # top left
        cv2.line(mask_neon, (x1, y1),(x1 + l,y1), outer_color, i)
        cv2.line(mask_neon, (x1, y1), (x1,y1 + l), outer_color, i)
        # top right
        cv2.line(mask_neon, (x2, y1), (x2 - l, y1), outer_color, i)
        cv2.line(mask_neon, (x2, y1), (x2, y1 + l), outer_color, i)
        # bottom left
        cv2.line(mask_neon, (x1, y2), (x1 + l, y2), outer_color, i)
        cv2.line(mask_neon, (x1, y2), (x1, y2 - l), outer_color, i)
        
        # bottom right
        cv2.line(mask_neon, (x2, y2), (x2 - l,y2), outer_color, i)
        cv2.line(mask_neon, (x2, y2), (x2,y2 - l), outer_color, i)
        # at certain pixel, it will look at surrounding (x,y) the bigger x and y are, the more blurred it will be 
        mask_neon = cv2.GaussianBlur(mask_neon,(2*i +1, 2*i + 1), 0)
    # Step 3 - Blend with the image
    # why it need to be one?
    
    cv2.addWeighted(mask_neon,0.7, image, 1,0,image)
    # cv2.addWeighted(mask_neon,1, image, 1,0,image)
    # Step 4 - write the usual white text
    
    # Top left
    cv2.line(image, (x1, y1), (x1 + l,y1), inner_color,thick)
    cv2.line(image, (x1, y1), (x1,y1 + l), inner_color,thick)
    
    # Top Right
    cv2.line(image, (x2, y1), (x2 - l, y1), inner_color,thick)
    cv2.line(image, (x2, y1), (x2, y1 + l), inner_color,thick)
    # Bottom Left
    cv2.line(image, (x1, y2), (x1 + l, y2), inner_color,thick)
    cv2.line(image, (x1, y2), (x1, y2 - l), inner_color,thick)
    # Bottom Right
    cv2.line(image, (x2, y2), (x2 - l,y2), inner_color,thick)
    cv2.line(image, (x2, y2), (x2,y2 - l), inner_color,thick)
    

    end = time.time()
    print(f"Corner took {end-start}")


def draw_neon_label(image, x1, y1,x2, y2, label, 
                    inner_text_color,
                    outer_text_color, 
                    text_size,
                    bg_box_color,
                    inner_border_color,
                    outer_border_color,
                    neon_strength,
                    border_thick):
    
    """
    draw_neon_label: this draw the label, fill the label rectangle, and write the text. Note the code is kinda messed up 
    due to the optimization.
    image: the origional image
    (x1,y1): Top left of the detected object
    (x2,y2): bottom right of the detected object
    inner_text_color, outer_text_color: inner and outer glow colors for text, respectively
    bg_box_color: the background fill of the label rectangle
    inner_border_color, outer_border_color: inner and outer glow colors for border line, respectively
    neon_strength: how far the spreading light 
    border_thick: the thickness of the border
    return: No return
    """
    
    
    style_text = cv2.FONT_HERSHEY_SIMPLEX
    Name = f"{label["name"]}"
    Job = f"{label["job"]}"
    Country = f"{label["country"]}"
    text_width, text_height = get_width_height_text(Name, Job, Country, text_size)    
    line_spacing = 4
    padding = 8
    connector_length = 20
    
    rec_width = text_width + 2 * padding
    rec_height = 3 * text_height + 2 * padding + 2 * line_spacing
    box_point, rect_point, bottom_left = get_points_for_connector(
                                                    image, 
                                                    x1,
                                                    y1,
                                                    x2,
                                                    y2, 
                                                    connector_length,
                                                    rec_width,
                                                    rec_height)

    
    

    name_start, job_start, country_start = get_points_for_text(bottom_left,
                                                               text_height,
                                                               line_spacing,
                                                               padding)
    # ------------ Fill in the label box -----------------
    # Step 1: create new layer
    overlay = image.copy()
    # Step: fill the color in that new layer
    cv2.rectangle(overlay,
                  (bottom_left[0], bottom_left[1] - rec_height),
                  (bottom_left[0] + rec_width, bottom_left[1]), 
                  bg_box_color, -1)
    # Step 3: Blend the image. Remember alpha and beta need to add up 1 to create 
    # effect one above other
    cv2.addWeighted(overlay,0.6,image,1- 0.6,0,image) 
    # ----------------------------------------------------
    
    
    # -----------------Draw neon corners and label--------
    # Step 1: Make a mask
    mask_neon = image.copy()
    mask_neon[:] = (0,0,0) # didn't affect anything if blend
    # Step 2: Apply blurred text, the bigger the more blurred
    for i in range(border_thick + neon_strength, 1, -1):
        # rectangle
        cv2.rectangle(mask_neon, 
                      (bottom_left[0], bottom_left[1] - rec_height), 
                      (bottom_left[0] + rec_width, bottom_left[1]), 
                      outer_border_color, i)
        # line connector
        cv2.line(mask_neon, box_point,rect_point, outer_border_color, i) 
        # text
        cv2.putText(mask_neon,Name, name_start,style_text, text_size ,outer_text_color, i)
        cv2.putText(mask_neon,Job, job_start,style_text, text_size ,outer_text_color, i)
        cv2.putText(mask_neon,Country, country_start,style_text, text_size ,outer_text_color, i)
        
        # blur elements
        mask_neon = cv2.GaussianBlur(mask_neon,(2*i +1, 2*i + 1), 0)
    # Step 3 - Blend with the image
    cv2.addWeighted(mask_neon,0.7, image, 1,0,image)
    # Step 4 - write the usual white text
    cv2.rectangle(image, 
                  (bottom_left[0], bottom_left[1] - rec_height),
                  (bottom_left[0] + rec_width, bottom_left[1]), 
                  inner_border_color,
                  1)
    cv2.line(image, box_point,rect_point, inner_border_color, 1)
    cv2.putText(image,Name, name_start,style_text, text_size ,inner_text_color, 1)
    cv2.putText(image,Job, job_start,style_text, text_size ,inner_text_color, 1)
    cv2.putText(image,Country, country_start,style_text, text_size ,inner_text_color, 1)
