# Hae Lee Kim
# Zhengliang Liu
# ENPM673 Project 5

import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from ReadCameraModel import *
from UndistortImage import *
import csv

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('C:/Users/Zheng/ENPM673/Oxford_dataset/model/')
K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
H = np.identity(4)
# print(K)
#img_green=cv2.imread("Green/green_frame%d.jpg" %i)
i = 1
# img_red=cv2.imread("img/frame%d.jpg" %i)

for i in range (740,3872):
	j =i+1
	img_1=cv2.imread("img/frame%d.jpg" %i,0)
	img_2=cv2.imread("img/frame%d.jpg" %j,0)
	# img1 = cv2.resize(img_1,(0,0),fx=0.5,fy=0.5)
	# img2 = cv2.resize(img_2,(0,0),fx=0.5,fy=0.5)

	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img_1,None)
	kp2, des2 = sift.detectAndCompute(img_2,None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	good = []
	pts1 = []
	pts2 = []

	for i,(m,n) in enumerate(matches):
		# if m.distance < 0.5*n.distance:
		good.append(m)
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	# for j in range (8):
	# 	firstimg.append(random.choice(pts1))
	# 	secondimg.append(random.choice(pts2))
	# firstimg=np.vstack(firstimg)
	# secondimg=np.vstack(secondimg)
	# F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
	E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx,cy), method=cv2.RANSAC, prob=0.999, threshold=0.5)
	# print('E_built', E)
	# E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
	# A = EstimateFundamentalMatrix(firstimg,secondimg)
	# We select only inlier points
	# pts1 = pts1[mask.ravel()==1]
	# pts2 = pts2[mask.ravel()==1]
	_, cur_R, cur_t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx,cy))
	if np.linalg.det(cur_R)<0:
		cur_R = -cur_R
	# if np.linalg.det(cur_t)<0:
	# 	cur_t = -cur_t
	new_pose = np.hstack((cur_R, cur_t))
	new_pose = np.vstack((new_pose,np.array([0,0,0,1])))
	# print(cur_R)
	# print(cur_t)
	x1 = (H[0][3])
	z1 = (H[2][3])
	H = H@new_pose
	x = (H[0][3])
	z = (H[2][3])
	# print((H[1][3]),'yyyy')

	img1 = cv2.resize(img_1,(0,0),fx=0.5,fy=0.5)
	cv2.imshow('frame',img1)
	# plt.plot(0,0)
	# plt.scatter(x,z)

	# Create CSV  and save data
	with open('built_in_data_all.csv','a', newline='') as f:
		writer=csv.writer(f)
		writer.writerow([x1, x, -z1, -z])
	# print((x1, x, -z1, -z))

	plt.plot([x1,x],[-z1,-z],'o')

	plt.pause(0.001)
	cv2.waitKey(1)