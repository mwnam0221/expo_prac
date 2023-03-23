
import glob
import numpy as np
import imutils
import cv2

image_paths=glob.glob('/home/nam/바탕화면/sadat/depth/*.png')
images = []


for image in image_paths:
    img = cv2.imread(image)
    images.append(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)

imageStitcher = cv2.Stitcher_create()
error, stitched_img = imageStitcher.stitch(images)

if not error:

    cv2.imwrite('stitched_depth.png', stitched_img)
    cv2.namedWindow(str('delta'), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    cv2.imshow('delta', stitched_img)
    cv2.waitKey(0)
    
