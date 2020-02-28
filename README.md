# Perception-for-Autonomous-Robots
ENPM6733Perception for Autonomous RobotsImage Processing and Computer Vision techniques for Mobile Robots is taught. Three topics are covered: Image Processing (Image Enhancement, Filtering, Advanced Edge and Texture ), 3D Vision (3D Geometry from Multiple view geometry, Motion Processing and Stereo) and an Introduction to Image Segmentation and Object Recognition. Students are introduced to a number of existing software toolboxes from Vision and Robotics, and will implement a number of smaller projects in OpenCV.RoboticsCore

### 1. Trafic Signs Detection [Github Link](https://github.com/ChoLiu/Perception-for-Autonomous-Robots/tree/master/Traffic%20Signs%20Detection)
In this project we aim to do Traffic Sign Recognition. You will perform the two steps of detection and recognition.
You can use existing OpenCV code (HOG feature detector, MSER feature detector, SVM routines) to create
the complete pipeline. The challenge will be in tuning the system to detect well.

The link of image data set is in the ENPM673_P6.pdf file. The project guideline is also in this file.

Result
<img src= "Video_Results/detection.gif" width="1000" height="500" >

### 2. Visual Odometry [Github Link](https://github.com/ChoLiu/Perception-for-Autonomous-Robots/tree/master/Visual%20Odometry)
Visual Odometry is a crucial concept in Robotics Perception for estimating the trajectory of the robot (the
camera on the robot to be precise). The concepts involved in Visual Odometry are quite the same for SLAM
which needless to say is an integral part of Perception.
In this project you are given frames of a driving sequence taken by a camera in a car, and the scripts
to extract the intrinsic parameters. Your will implement the different steps to estimate the 3D motion of the
camera, and provide as output a plot of the trajectory of the camera

The link of image data set is in the ENPM673_P6.pdf file. The project guideline is also in this file.

Result (Note:Our own function took about 24hr to run.)

The blue plot shows the build-in function and the orange line shows the output of our own function.
<img src= "Video_Results/Visual Odometry.gif" width="1000" height="500" >
