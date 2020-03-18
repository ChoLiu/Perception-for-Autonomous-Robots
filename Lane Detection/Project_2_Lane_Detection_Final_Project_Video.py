# Hae Lee Kim
# Zhengliang Liu
# ENPM673 Lane Detection

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt

def findH():
    # Points used for Homography 
    Road_pt = [(600,450),(685,450),(320,675),(1090,675)]#Source coordinates
    Ref_pt = [(100,0),(500,0),(100,720),(500,720)]#destination coordinates
    H, mask = cv2.findHomography(np.array(Road_pt), np.array(Ref_pt))#Source, destination

    Road_pt = [(600,450),(685,450),(320,675),(1090,675)]#Source coordinates
    Ref_pt_1 = [(320,0),(400,0),(320,720),(400,720)]#destination coordinates
    #[(320,0),(400,0),(320,720),(400,720)]
    H_1, mask = cv2.findHomography(np.array(Road_pt), np.array(Ref_pt_1))
    return H,H_1
# Undistortion, Color Thresholding
def processimage(frame):

    #Image undistortion
    K = np.array([[1.15422732e+03,0,  6.71627794e+02], [0, 1.14818221e+03, 3.86046312e+02], [0, 0, 1.00000000e+00]])
    dist = np.array([ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05, 2.20573263e-02])
     
    crop_pt = np.array([[0,0],[1280,0],[1280,400],[0,400]], 'int32')
    frameundistort = cv2.undistort(frame, K, dist)
    croped_sky = cv2.fillPoly(frame, pts =[crop_pt], color=0)
    

    # Unwarp using homography
    gray_frame = cv2.cvtColor(croped_sky, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    # gray_frame_dst = cv2.warpPerspective(gray_frame, H, (720, 800))
    color_frame_dst = cv2.warpPerspective(frame, H, (720, 800))
    color_frame_large = cv2.warpPerspective(frame, H_reg, (720, 800))

    # HSV mask setting that detects yellow on shitty road
    # hsv1 = cv2.cvtColor(color_frame_large, cv2.COLOR_RGB2HSV)
    hsv1 = cv2.GaussianBlur(color_frame_large, (5, 5), 0)
    

    white_lower = np.array([0,200,0])
    white_upper = np.array([255,255,255])
    white_mask = cv2.inRange(hsv1,white_lower,white_upper)

    yellow_hsv_low  = np.array([20, 75, 150])
    yellow_hsv_high = np.array([130, 255, 255])
    yellow_mask = cv2.inRange(hsv1, yellow_hsv_low, yellow_hsv_high)

    mask = cv2.bitwise_or(yellow_mask,white_mask)
    masked_top = cv2.bitwise_and(color_frame_large, color_frame_large, mask = mask)
    
    #Test HSV
    #HSV mask setting that detects yellow on shitty road
    hsv = cv2.cvtColor(color_frame_dst, cv2.COLOR_RGB2HSV)
    hsv = cv2.GaussianBlur(color_frame_dst, (5, 5), 0)
    
    white_lower = np.array([0,200,0])
    white_upper = np.array([255,255,255])
    white_mask = cv2.inRange(hsv,white_lower,white_upper)

    yellow_hsv_low  = np.array([20, 75, 150])
    yellow_hsv_high = np.array([130, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_hsv_low, yellow_hsv_high)

    mask = cv2.bitwise_or(yellow_mask,white_mask)
    masked_top_zoomed = cv2.bitwise_and(color_frame_dst, color_frame_dst, mask = mask)

    return color_frame_dst,masked_top_zoomed,masked_top

# Polyfit
# Non-zoomed polyfit
def fit_polyline1(binary_warped):
    # Take a histogram of the bottom half of the image
    binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY) 
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    x_left = np.where(histogram == histogram.max())[0][0]
    
    peaks, _ = find_peaks(histogram, prominence=1) 
    peaks.sort() 
    x_right=400

    # Find the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    white_pixels_y = np.array(nonzero[0])
    white_pixels_x = np.array(nonzero[1])
   
    # Set the width of the windows +/- margin
    wid = 20
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    left_LL_bound = x_left - wid
    right_LL_bound = x_left + wid
    left_RL_bound = x_right - wid
    right_RL_bound = x_right + wid

    good_left_inds = ((white_pixels_x >= left_LL_bound) & (white_pixels_x < right_LL_bound) & (white_pixels_y < 800) & (white_pixels_y >= 100)).nonzero()[0]
    good_right_inds = ((white_pixels_x >= left_RL_bound) & (white_pixels_x < right_RL_bound) & (white_pixels_y < 800) & (white_pixels_y >= 100)).nonzero()[0]
    
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
        

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    #print(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx1 = white_pixels_x[left_lane_inds]
    lefty1 = white_pixels_y[left_lane_inds]
    rightx1 = white_pixels_x[right_lane_inds]
    righty1 = white_pixels_y[right_lane_inds]
    

    # Fit a second order polynomial to each
    left_fit1 = np.polyfit(lefty1, leftx1, 2)
    right_fit1 = np.polyfit(righty1, rightx1, 2)

    # Find the fitted curve
    plot_x = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx1 = left_fit1[0]*plot_x**2 + left_fit1[1]*plot_x + left_fit1[2]
    right_fitx1 = right_fit1[0]*plot_x**2 + right_fit1[1]*plot_x + right_fit1[2]
    
    # Creating a empty img for curves
    color_warp1 = np.zeros((1280,720,3),np.uint8)
    
    return left_fitx1,right_fitx1,color_warp1,plot_x,leftx1,lefty1,rightx1,righty1

def fit_polyline(binary_warped):
    # Take a histogram of the bottom half of the image
    binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY) 
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    x_left = np.where(histogram == histogram.max())[0][0]
    peaks, _ = find_peaks(histogram, prominence=1) 
    peaks.sort() 
    x_right=peaks[len(peaks)-1]
    out = []
    out1 = []
    if x_right >600:
        out.append(550)
        x_right = out[0]
    if x_left >200:
        out1.append(95)
        x_left = out1[0]
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    white_pixels_y = np.array(nonzero[0])
    white_pixels_x = np.array(nonzero[1])
   
    # Set the width of target area
    wid = 100

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    left_LL_bound = x_left - wid +50
    right_LL_bound = x_left + wid
    left_RL_bound = x_right - wid
    right_RL_bound = x_right + wid

    good_left_inds = ((white_pixels_x >= left_LL_bound) & (white_pixels_x < right_LL_bound) & (white_pixels_y < 800) & (white_pixels_y >= 0)).nonzero()[0]
    good_right_inds = ((white_pixels_x >= left_RL_bound) & (white_pixels_x < right_RL_bound) & (white_pixels_y < 800) & (white_pixels_y >= 0)).nonzero()[0]
    
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
        

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    #print(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = white_pixels_x[left_lane_inds]
    lefty = white_pixels_y[left_lane_inds]
    rightx = white_pixels_x[right_lane_inds]
    righty = white_pixels_y[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Find the fitted curve
    plot_x = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*plot_x**2 + left_fit[1]*plot_x + left_fit[2]
    right_fitx = right_fit[0]*plot_x**2 + right_fit[1]*plot_x + right_fit[2]
    
    # Creating a empty img for curves
    color_warp = np.zeros((800,720,3),np.uint8)
    
    return left_fit,right_fit,left_fitx,right_fitx,color_warp,plot_x,leftx,lefty,rightx,righty


# Creat a plot for histogram
fig, ax = plt.subplots()
ax.set_xlim(0, 720)
ax.set_ylim(0, 800)
line, = ax.plot(np.arange(720), np.zeros((720,1)), c='b', lw=1)
plt.ion()
plt.show()

try: 
    cap = cv2.VideoCapture('project_video.mp4')
    while(cap.isOpened()):

        ret, frame = cap.read()
        ret, final_out = cap.read()

        H,H_reg = findH()

        color_frame_dst,masked_top_zoomed,masked_top = processimage(frame)
        Unwarped_frame_zoom = cv2.warpPerspective(frame, H, (720, 800))
        Unwarped_frame = cv2.warpPerspective(frame, H_reg, (720, 800))
        cv2.imshow('reg',Unwarped_frame)
        cv2.imshow('zoom',Unwarped_frame_zoom)

        Road_pt = [(600,450),(685,450),(320,675),(1090,675)]#Source coordinates
        Ref_pt = [(100,0),(500,0),(100,720),(500,720)]#destination coordinates
        curve_pt = [(0,0),(800,0),(800,720),(0,720)]

        M = cv2.getPerspectiveTransform(np.float32(Ref_pt), np.float32(Road_pt))

        _,_,left_fitx,right_fitx,color_warp,plot_x,leftx,lefty,rightx,righty = fit_polyline(masked_top_zoomed)
        left_fitx1,right_fitx1,color_warp1,plot_x,leftx1,lefty1,rightx1,righty1= fit_polyline1(masked_top)

        pts_left = np.array([np.transpose(np.vstack([left_fitx, plot_x]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plot_x])))])
        
        # Generate the pixces witin the curves
        pts = np.hstack((pts_left, pts_right))
        pts_left1 = np.array([np.transpose(np.vstack([left_fitx1, plot_x]))])
        pts_right1 = np.array([np.flipud(np.transpose(np.vstack([right_fitx1, plot_x])))])

        pts1 = np.hstack((pts_left1, pts_right1))

        #live histogram
        gray = cv2.cvtColor(masked_top_zoomed, cv2.COLOR_BGR2GRAY)
        histogram = np.sum(gray[gray.shape[0]//2:,:], axis=0)
        line.set_ydata(histogram//100)
        fig.canvas.draw()

        #Plot line fit
        color_countours = np.zeros((800,720,3),np.uint8)
        color_countours1 = np.zeros((800,720,3),np.uint8)

        color_countours = cv2.drawContours(color_countours, np.int_([pts]), -1, (255, 255, 0), 15)
        # color_countours_reg = cv2.drawContours(color_countours1, np.int_([pts1]), -1, (255, 255, 0), 15)#Just for curvature calculation
        #Plot front area within fitted area
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 255))
        
        #Convert topview to video 
        newwarp = cv2.warpPerspective(color_warp, M,(1280,720))
        newwarp_countours=cv2.warpPerspective(color_countours, M,(1280,720))

        result  = cv2.addWeighted(final_out, 1, newwarp_countours, 0.6, 0)
        result = cv2.addWeighted(result, 1, newwarp, 0.3, 0)
        
        # result_reg =cv2.addWeighted(masked_top, 1, color_countours_reg, 0.6, 0)
        masked_top= cv2.resize(masked_top, (0, 0),fx=1,fy=1)
        result = cv2.resize(result, (0, 0),fx=1,fy=1)
        frame = cv2.resize(frame, (0, 0),fx=1,fy=1)
        Unwarped_frame = cv2.resize(Unwarped_frame, (0, 0),fx=1,fy=1)

        # Calculate radius of curvature 
        x_pix_meters = 3.7/80 #3.7m/80 pixels
        left_fit1 = np.polyfit(lefty1, leftx1, 2)
        # left lane 
        L_x_eval = 720
        L_first_derv = 2*left_fit1[0]*L_x_eval+left_fit1[1]
        L_second_derv = 2*left_fit1[0] 
        L_curvature = (((1+(L_first_derv)**2))**1.5)/np.absolute(L_second_derv)
        L = (((1+(L_first_derv)**2))**1.5)/(L_second_derv)
        L_curve_radius_m = L_curvature*x_pix_meters

        if L_curve_radius_m > 1500:
            cv2.putText(result, "Straight" ,(10, 100),cv2.FONT_HERSHEY_SIMPLEX ,2,(225,255,0),2)

        if L_curve_radius_m < 1000 and L < 0:
            cv2.putText(result, "Turning Left" ,(10, 100),cv2.FONT_HERSHEY_SIMPLEX ,2,(225,255,0),2)

        if L_curve_radius_m < 1300 and L > 0:
            cv2.putText(result, "Turning Right" ,(10, 100),cv2.FONT_HERSHEY_SIMPLEX ,2,(225,255,0),2)

        # Display radius on frame
        cv2.putText(result, "Radius = %s m" %L_curve_radius_m,(10, 50),cv2.FONT_HERSHEY_SIMPLEX ,2,(225,255,0),2)

        cv2.imshow('Final out put',result )

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
except:
    print("Video Complete")
