'''
Author: Minh Long Vu
Application: This functions draw the customized bounding box instead of traditional rectangle one
'''

import cv2


def draw_corners(image, x1, y1, x2, y2, length, thick):
    
    '''
    draw_corners: Basically this draw the side of rectangle but instead of one full side from corner
    to corner. It only draw two short line from a corner
    image: an numpy array representing image
    x1: horizontal position of top right corner point (x1,y1)
    y1: vertical position of top right corner point (x1,y1)
    x2: horizontal position of top right corner point (x2,y2)
    y2: vertical position of top right corner point (x2,y2)
    length: the length of that short line. If you specify x2 - x1 then it is basically
    traditional side
    thick: the thickness of each line
    return: No return
    '''
    # rename for shorter word
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
    
    '''
    draw_label: this draw the label together with rectangle surrounding it. The reason why
    we only take (X1,y1) is because we want to draw on the top. You could customize by using
    (x2,y2) if you want to draw in the bottom
    image: an numpy array representing image
    x1 - Int: horizontal position of top right corner point (x1,y1)
    y1 - Int: vertical position of top right corner point (x1,y1)
    label - String: What text do you want to display as a label for the object
    color_text - (Int,Int,Int): tuple of three RGB specifying the color of label
    color_box - (Int,Int,Int): tuple of three RGB specifying the color of label
    '''
    
    
    overlay = image.copy()
    # We want to fit the label inside the rectangle so we need to know the size of label

    (text_width, text_height), _ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    
    # we need to use blending techquie: combine many images together 
    # if we want to add transparent rectangle
    cv2.rectangle(overlay,(x1 - 5,y1 - 15 + 10 ),(x1 + text_width + 5,y1 - text_height - 15 - 10), color_box,-1)
    # dst = alpha * overlay + beta * image + gamma
    # dst: write to where
    # addWeighted(overlay, alpha, image, beta, gamma, dst)
    cv2.addWeighted(overlay,0.6,image,1- 0.6,0,image) 
    # Note that if we want to make it feel like on top. the coffeificent should larger than
    # 0.5 which means it is more dense
    cv2.putText(image,label, (x1, y1 - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.8,  color_text, 1)