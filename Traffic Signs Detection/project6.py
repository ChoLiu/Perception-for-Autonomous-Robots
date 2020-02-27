# Hae Lee Kim
# Zhengliang Liu
# ENPM673 Project 6

import cv2
import numpy as np
import glob
import argparse
import os
import glob
import sys
import logging
print("11111111111")
# HOG
hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)

# Empty datasets 
train_data = []
test_data = []
label_test = []
label_train = []
video_test = []
imgcrop = np.zeros((1,1))

train_path = glob.glob("C:/Users/Zheng/ENPM673/Project6/TSR/Training/*/*.ppm")
# test_path = glob.glob("C:/Users/haele/Desktop/Documents/Spring2019/ENPM673/Project 6/TSR/Testingsign/00014/*.ppm")
print("Images Loaded")
# Run HOG on training data
for path in train_path:
    label = int(path.split("\\")[1])
    # print(label)
    img = cv2.imread(path)
    img = cv2.resize(img, (64,64))
    descriptors = hog.compute(img)
    train_data.append(descriptors)
    label_train.append(label)


train_data = np.float32(train_data)
train_data = np.resize(train_data, (len(train_data), 1764)) # 1764 is num of columns for train and test data
label_train = np.array(label_train).reshape(-1,1)

# SVM Model
svm = cv2.ml.SVM_create()
# Set SVM type
svm.setType(cv2.ml.SVM_C_SVC)
# Set SVM Kernel to Radial Basis Function (RBF) 
svm.setKernel(cv2.ml.SVM_LINEAR)
# Set parameter C
# svm.setC(0.5)#red
svm.setC(1)
# Set parameter Gamma
svm.setGamma(0)
svm.train(train_data, cv2.ml.ROW_SAMPLE,label_train)
svm.save("digits_svm_model.yml")

print('SVM done')

# Blue mask function
def bluemask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    blue_lower = np.array([6,75,20])
    blue_upper = np.array([20,255,255])
    ###balck and white
    # blue_lower = np.array([0,0,0])
    # blue_upper = np.array([255,255,255])
    blue_mask = cv2.inRange(hsv,blue_lower,blue_upper)
    mask = cv2.bitwise_or(blue_mask,blue_mask)
    blue = cv2.bitwise_and(img, img, mask = mask)
    return blue
    
# Red mask function
def redmask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #lower red
    lower_red = np.array([114,50,60])
    upper_red = np.array([200,150,255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    red = cv2.bitwise_and(img,img, mask= mask_red)
   
    return red

# Blue signs classification
def bluesign(img,ID,center,x,y,w,h):
    
    ycoor = abs(int(center[1])-8) #y
    xcoor = abs(int(center[0])-70) #x

    if ID == 35:
        img_logo = cv2.imread("directionsign.jpg", cv2.IMREAD_COLOR)
        img_logo = cv2.resize(img_logo,(64,64))
        cv2.rectangle(img, (x, y), (x+w+5, y+h+5), (0, 255, 0), 2) 
        roiImg = img_logo

        y =  (roiImg.shape)[0]
        x =  (roiImg.shape)[1]                         
        img[ycoor:ycoor+y,xcoor:xcoor+x] = roiImg
        
        # img = cv2.circle(img,(center[0],center[1]),30,(255,0,255),thickness = 3)              
    if ID == 38:
        img_logo = cv2.imread("bikesign.png", cv2.IMREAD_COLOR)
        img_logo = cv2.resize(img_logo,(64,64))
        cv2.rectangle(img, (x, y), (x+w+5, y+h+5), (0, 255, 0), 2)
        roiImg = img_logo
        y =  (roiImg.shape)[0]
        x =  (roiImg.shape)[1]                       
        img[ycoor:ycoor+y,xcoor:xcoor+x] = roiImg
        
        # img = cv2.circle(img,(center[0],center[1]),30,(255,0,255),thickness = 3)
    if ID == 45:
        img_logo = cv2.imread("parking.jpg", cv2.IMREAD_COLOR)
        img_logo = cv2.resize(img_logo,(64,64))
        cv2.rectangle(img, (x, y), (x+w+5, y+h+5), (0, 255, 0), 2)
        roiImg = img_logo
        y =  (roiImg.shape)[0]
        x =  (roiImg.shape)[1]                       
        img[ycoor:ycoor+y,xcoor:xcoor+x] = roiImg
        
        # img = cv2.circle(img,(center[0],center[1]),30,(255,0,255),thickness = 3) 

# Red signs classification   
def redsign(img,ID,center,x,y,w,h):
    
    ycoor = abs(int(center[1])-8) #y
    xcoor = abs(int(center[0])-70) #x

    if ID == 1:
        img_logo = cv2.imread("bumpsign.jpg", cv2.IMREAD_COLOR)
        img_logo = cv2.resize(img_logo,(64,64))
        cv2.rectangle(img, (x, y), (x+w+5, y+h+5), (0, 255, 0), 2)
        roiImg = img_logo
        y =  (roiImg.shape)[0]
        x =  (roiImg.shape)[1]              
        img[ycoor:ycoor+y,xcoor:xcoor+x] = roiImg
        # img = cv2.circle(img,(center[0],center[1]),30,(255,0,255),thickness = 3)
    if ID == 14:
        img_logo = cv2.imread("narrowsign.jpg", cv2.IMREAD_COLOR)
        img_logo = cv2.resize(img_logo,(64,64))
        cv2.rectangle(img, (x, y), (x+w+5, y+h+5), (0, 255, 0), 2)
        roiImg = img_logo
        y =  (roiImg.shape)[0]
        x =  (roiImg.shape)[1]                  
        img[ycoor:ycoor+y,xcoor:xcoor+x] = roiImg
        # img = cv2.circle(img,(center[0],center[1]),30,(255,0,255),thickness = 3)
    if ID == 17:
        img_logo = cv2.imread("rocketsign.png", cv2.IMREAD_COLOR)
        img_logo = cv2.resize(img_logo,(64,64))
        cv2.rectangle(img, (x, y), (x+w+5, y+h+5), (0, 255, 0), 2)
        roiImg = img_logo
        y =  (roiImg.shape)[0]
        x =  (roiImg.shape)[1]                      
        img[ycoor:ycoor+y,xcoor:xcoor+x] = roiImg
        # img = cv2.circle(img,(center[0],center[1]),30,(255,0,255),thickness = 3)
    if ID == 19:
        img_logo = cv2.imread("trisign.png", cv2.IMREAD_COLOR)
        img_logo = cv2.resize(img_logo,(64,64))
        cv2.rectangle(img, (x, y), (x+w+5, y+h+5), (0, 255, 0), 2)
        roiImg = img_logo
        y =  (roiImg.shape)[0]
        x =  (roiImg.shape)[1]                  
        img[ycoor:ycoor+y,xcoor:xcoor+x] = roiImg
        # img = cv2.circle(img,(center[0],center[1]),30,(255,0,255),thickness = 3)
    if ID == 21:
        img_logo = cv2.imread("stop.jpg", cv2.IMREAD_COLOR)
        img_logo = cv2.resize(img_logo,(64,64))
        cv2.rectangle(img, (x, y), (x+w+5, y+h+5), (0, 255, 0), 2)
        roiImg = img_logo
        y =  (roiImg.shape)[0]
        x =  (roiImg.shape)[1]                        
        img[ycoor:ycoor+y,xcoor:xcoor+x] = roiImg
        # img = cv2.circle(img,(center[0],center[1]),30,(255,0,255),thickness = 3)
             

# Input image path    
img_path = glob.glob("C:/Users/Zheng/ENPM673/Project6/TSR/input/*.jpg")

# Forward, Reverse, functions reference from Rene Jacques
i = 0
# for path in img_path:
fast_forward = False
rewind = False

f = 32640
while f < 35500:
    path = img_path[f-32640]
    keypress = cv2.waitKey(1)
    if keypress == ord('d'):
        if f+1 < 35500:
            f += 1
        # print(f-32640)
        rewind = False
        fast_forward = False
    elif keypress == ord('a'):
        if f-1 >= 32640:
            f -= 1
        # print(f-32640)
        rewind = False
        fast_forward = False
    elif keypress == ord('w'):
        fast_forward = True
        rewind = False
    elif keypress == ord('s'):
        rewind = True
        fast_forward = False
    else:
        if fast_forward:
            if f+1 < 35500:
                f += 1
                # print(f-32640)
        elif rewind:
            if f-1 >= 32640:
                f -= 1
                # print(f-32640)
        else:
            pass

    # img=cv2.imread('image.032729.jpg',cv2.IMREAD_COLOR)
    img=cv2.imread(path,cv2.IMREAD_COLOR)
    # img = cv2.imread('image.035134.jpg',cv2.IMREAD_COLOR)
    blue = bluemask(img)
    red = redmask(img)

    h, s, v1 = cv2.split(red)
    h1, s1, v2 = cv2.split(blue)

    ################################################################################
    out_binary, contours, hierarchy = cv2.findContours(v2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # Find contours to create bounding box for detection - Blue
    for cnt in range(len(contours)):
        
        epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
        approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
        
        area = cv2.contourArea(contours[cnt])
        
        # cv2.drawContours(img,contours,cnt, (0, 255, 0), 2)
        if area >1100:
            # print(len(approx))
            x, y, w, h = cv2.boundingRect(contours[cnt])
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            imgcrop = img[y:y+h,x:x+w]
            imgcrop = cv2.resize(imgcrop, (64,64))                  
            cv2.imshow('blue',imgcrop)
            cv2.moveWindow("blue",100,100)
            descriptors = hog.compute(imgcrop)
            video_test = [descriptors]
            test_data = np.resize(video_test, (len(video_test), 1764))
            
            center = (int(x),int(y))
            
            testResponse = svm.predict(test_data)[1].ravel()
            ycoor = abs(int(center[1])-10) #y
            xcoor = abs(int(center[0])-80)
            if ycoor + 64 > 1236 or xcoor+64>1628:
                continue
            bluesign(img,testResponse,center,x,y,w,h)
    
    ################################################################################
    out_binary, contours, hierarchy = cv2.findContours(v1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #  Find contours to create bounding box for detection - Red
    for cnt in range(len(contours)):
        
        epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
        approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
        
        area = cv2.contourArea(contours[cnt])
        
        # cv2.drawContours(img,contours,cnt, (0, 255, 0), 2)
        if area >900:
            # print(len(approx))
            x, y, w, h = cv2.boundingRect(contours[cnt])
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            imgcrop = img[y:y+h+5,x:x+w+5]
            imgcrop = cv2.resize(imgcrop, (64,64))
            cv2.imwrite("frame%d.ppm" %i, imgcrop)                   
            cv2.imshow('red',imgcrop)
            cv2.moveWindow("red",100,300)
            descriptors = hog.compute(imgcrop)
            video_test = [descriptors]
            test_data = np.resize(video_test, (len(video_test), 1764))

            center = (int(x),int(y))
            ycoor = abs(int(center[1])-10) #y
            xcoor = abs(int(center[0])-80)
            if ycoor + 64 > 1236 or xcoor+64>1628:
                continue

            testResponse = svm.predict(test_data)[1].ravel()

            if h/w < 3 and w/h<1.3:
                redsign(img,testResponse,center,x,y,w,h)
    #################################################################################  
    masked_top = cv2.resize(v1,(0,0),fx=0.5,fy=0.5)
    img=cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    cv2.imshow('22',masked_top)
    cv2.imshow('33',img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break