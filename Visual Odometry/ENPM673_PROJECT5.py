## Zhengiang Liu
## Hae Lee Kim
## EMPM673_PROJECT5
## 5/7/19

import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from ReadCameraModel import *
from UndistortImage import *
import csv

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('C:/Users/Zheng/ENPM673/Oxford_dataset/model/') #Change directory
K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])
B = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0]])
H = np.identity(4)

class Inlier():
	def __init__(self, match, position):
		self.match = match
		self.position = position

def random_points(points1,points2):
	pts_index = np.random.randint(len(points1), size=8)
	X1 = []
	X2 = []

	for i in range(8):
		X1.append(points1[pts_index[i]])
		X2.append(points2[pts_index[i]])

	return np.array(X1), np.array(X2)

def EstimateFundamentalMatrix(source, dst):
	
	variantion_x1 = np.mean(source[:,0])
	variantion_y1 = np.mean(source[:,1])
	variantion_x2 = np.mean(dst[:,0])
	variantion_y2 = np.mean(dst[:,1])

	x1_dis = (source[:,0]-variantion_x1)**2
	y1_dis = (source[:,1]-variantion_y1)**2
	x2_dis = (dst[:,0]-variantion_x2)**2
	y2_dis = (dst[:,1]-variantion_y2)**2

	distance_1 = np.sqrt(x1_dis+y1_dis)
	distance_2 = np.sqrt(x2_dis+y2_dis)

	ratio1 = np.sqrt(2)/np.mean(distance_1)
	ratio2 = np.sqrt(2)/np.mean(distance_2)

	# Combine camera and world arrays by horizontally concatenating
	img_1_2 = np.hstack((source, dst))

	# Iterate through each of the four points and form 2x9 matrix
	A_matrix = np.zeros((8,9))

	# print(np.shape(img_1_2)[0])
	for i in range(np.shape(img_1_2)[0]): 
		x = (source[i][0]-variantion_x1)*ratio1
		y = (source[i][1]-variantion_y1)*ratio1
		x_1 = (dst[i][0]-variantion_x2)*ratio2
		y_1 = (dst[i][1]-variantion_y2)*ratio2
		row = [x*x_1, x_1*y, x_1, y_1*x, y_1*y, y_1, x, y,1]

		
		A_matrix[i]=row
		
	# Compute SVD of A
	U, S, V = np.linalg.svd(A_matrix)

	# Compute h
	V = V[8]# Last column of V matrix from SVD

	#print(np.shape(h))
	F = np.reshape(V, (3, 3)) # Fundamental matrix
	UF,SF,VF = np.linalg.svd(F)

	SF[2] = 0
	SF = np.diag(SF)
	F = UF @ SF @ VF
	T1 =np.array([[ratio1,0,-ratio1*variantion_x1],[0,ratio1,-ratio1*variantion_y1],[0,0,1]])
	T2 =np.array([[ratio2,0,-ratio2*variantion_x2],[0,ratio2,-ratio2*variantion_y2],[0,0,1]])
	F = T2.T @ F @ T1
	F = F/F[2,2] 

	return F

def LinearTriangulation(K, C, R, C2, R2, x1, x2):
	P1 = K @ np.hstack((R, -C))
	P2 = K @ np.hstack((R2, -C2))
	P1= np.array(P1)
	P2= np.array(P2)
	N=len(x1)

	#Linear triangulation
	x_coor=[]
	z_coor=[]
	X=[]
	ch_list=[]
	check_number = 0
	for i in range (N):
		
		A = ([x1[i][0] * P1[2,:] - P1[0, :], x1[i][1] * P1[2, :] - P1[1, :], 
			x2[i][0] * P2[2, :] - P2[0, :], x2[i][1] * P2[2, :] - P2[1, :]])

		_,_,V = np.linalg.svd(A)
		V1 = V[3]# Last column of V matrix from SVD
		V1 = V1.reshape(4,1)
		x_check = V1/V1[3]
		ch= R2[2]@(x_check[0:3]-C2)
		if ch>0 and x_check[2]>0:
			check_number+=1

	return check_number

for l in range(0,3872):
	
	j =l+1
	img_1=cv2.imread("img/frame%d.jpg" %l,0)
	img_2=cv2.imread("img/frame%d.jpg" %j,0)
	# img1 = cv2.resize(img_1,(0,0),fx=0.5,fy=0.5)
	# img2 = cv2.resize(img_2,(0,0),fx=0.5,fy=0.5)

	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img_1,None)
	kp2, des2 = sift.detectAndCompute(img_2,None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	good = []
	pts1 = []
	pts2 = []

	for i,(m,n) in enumerate(matches):
		if m.distance < 0.5*n.distance:
			good.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)

	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	# print(pts1.shape)
	src_pts = np.vstack(pts1.reshape(-1,1,2))
	dst_pts = np.vstack(pts2.reshape(-1,1,2))
	
	# Restructure data 
	source_dst = np.hstack((src_pts,dst_pts)) # Data structure to be [x1,y1,x2,y2] of all correspondences
	
	# Start RANSAC Here
	M = 500 # num of iterations
	N = len(good) # Number of correspondences. Replace with sift correspondences
	inlier_thres = 0.8# Condition for percentage of desired inliers
	n = 0 # Number of inlie

	for i in range(M):
		if i%100 == 0:
			print(i)
		x,x_1 = random_points(pts1,pts2)
		F = EstimateFundamentalMatrix(x, x_1)
		s = [] # Set of inliers
		e = 0.1 # Threshold

		for j in range(N):
			#U1 IS FIRST IMAGE
			if F is None:
				continue
			first_vec = np.array([source_dst[j][0], source_dst[j][1], 1]).reshape(-1,1)
			second_vec = np.array([source_dst[j][2], source_dst[j][3], 1]).reshape(-1,1) #[x1, y2, 1]
			x2_transpose = np.transpose([source_dst[j][2], source_dst[j][3], 1]) #[x1, y2, 1]

			# Denominator
			Fx_11 = (np.dot(F, first_vec))[0]
			Fx_12 = (np.dot(F, first_vec))[1]
			Fx_21 = (np.dot(np.transpose(F), second_vec))[0]
			Fx_22 = (np.dot(np.transpose(F), second_vec))[1]
			thres_cond =((x2_transpose@F@first_vec)**2)/(Fx_11**2 + Fx_12**2 + Fx_21**2 + Fx_22**2)
			
			if thres_cond[0] <= e: # If within threshold for determining whether pair is inlier. e should be close to 0
				match = good[j]
				position = source_dst[j]
				inlier = Inlier(match, position)
				s.append(inlier) # Add the correspondences to set
		current_thres = len(s)/N 
	

	# Check for number of inlier threshold
		if n < len(s):
			n = len(s)
			s_final = s
			F_final = F

	B = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0]])
	E = np.transpose(K) @ F_final @ K
	U1,_,V1 = np.linalg.svd(E)
	E = U1@B@V1
	U,_,V = np.linalg.svd(E)

	# E = U @ D @ V
	####3.4####
	W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])

	c1 = np.array(U[:,2]).reshape(3,1)
	c2 = -np.array(U[:,2]).reshape(3,1)
	c3 = np.array(U[:,2]).reshape(3,1)
	c4 = -np.array(U[:,2]).reshape(3,1)

	r1 = U@W@V
	r2 = U@W@V
	r3 = U@W.T@V
	r4 = U@W.T@V
	
	if np.linalg.det(r1)<0:
		c1=-c1
		c2=-c2
		r1=-r1
		r2=-r2

	if np.linalg.det(r3)<0:
		c3=-c3
		c4=-c4
		r3=-r3
		r4=-r4
	
	# Put trans and rot in arrays
	trans=[c1,c2,c3,c4]
	rot =[r1,r2,r3,r4]
	check1=LinearTriangulation(K, np.zeros((3, 1)), np.identity(3), trans[0], rot[0], pts1, pts2)
	check2=LinearTriangulation(K, np.zeros((3, 1)), np.identity(3), trans[1], rot[1], pts1, pts2)
	check3=LinearTriangulation(K, np.zeros((3, 1)), np.identity(3), trans[2], rot[2], pts1, pts2)
	check4=LinearTriangulation(K, np.zeros((3, 1)), np.identity(3), trans[3], rot[3], pts1, pts2)
	z_set=np.array([[check1],[check2],[check3],[check4]])
	
	# Find max matches of  to choose C and R
	index = np.argmax(z_set)
	new_pose = np.hstack((rot[index], -trans[index]))
	new_pose = np.vstack((new_pose,np.array([0,0,0,1])))
	x1 = (H[0][3])
	z1 = (H[2][3])
	H = H @ new_pose
	x = (H[0][3])
	z = (H[2][3])

	# Create CSV  and save data
	# with open('x_z_data_newransac.csv','a', newline='') as f:
	# 	writer=csv.writer(f)
	# 	writer.writerow([x1, -z1, x, -z])

	img_1 = cv2.resize(img_1,(0,0),fx=0.5,fy=0.5)
	cv2.imshow('figure',img_1)
	plt.plot([x1,x],[-z1,-z],'o')
	plt.pause(0.01)
	cv2.waitKey(1)

cv2.waitKey(0)