'''
SelfDrivingCar is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SelfDrivingCar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SelfDrivingCar.  If not, see <https://www.gnu.org/licenses/>.
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
step1-
covert to grayScale

step2-
reduce noise, as noise can create false edges so we perform noise
reduction using Gausian filter using 5x5 kernel to reduce noise,
canny edge detection uses 5x5 gausian blur so we can later on remove this step

step3-
cv2.Canny(image,Low_Threshold,Upper_Threshold)

edge detection to find sharp change in the pixel intensities
computes gradient, if gradient is larger than upper threshold then it is accepted
if it is lower than lower threshold then it is rejected, if it is between the upper and lower then it is
accepted only if it is connected to a strong edge

step4-
mask and find region of interest

step5-
identify the lane lines by hough transform(identify the straight lines)
consider the line with maximum point of intersection in the hough space thus we can say that line of best fit

'''

def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
       # print(left_fit)
        #print(right_fit)
        left_fit_average=np.average(left_fit,axis=0)
        right_fit_average=np.average(right_fit,axis=0)
        #print(left_fit_average,'left')
        #print(right_fit_average,'right')

def canny(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            #print(line)
            x1,y1,x2,y2=line.reshape(4)# 4d array to 1d array
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image            
        




'''from matplotlib we find triangle from the graph'''  
def region_of_interest(image):
    height=image.shape[0]
    polygons=np.array([[(200,height),(1100,height),(550,250)]
    ])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image


'''
image=cv2.imread("test_image.jpg")
lane_image=np.copy(image)
canny_image=canny(lane_image)
cropped_image=region_of_interest(canny_image)
lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
#2 pixels followed by 1 degree precision followed by threshold,
#min no. of votes needed to accept a candidate line
#then we have a place holder array, minLine is anything below 40 pixels are rejected
averaged_lines=average_slope_intercept(lane_image, lines)
line_image=display_lines(lane_image,lines)
combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
cv2.imshow('result',combo_image)
cv2.waitKey(0)
'''


cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _,frame=cap.read()
    canny_image=canny(frame)
    cropped_image=region_of_interest(canny_image)
    lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    #2 pixels followed by 1 degree precision followed by threshold,
    #min no. of votes needed to accept a candidate line
    #then we have a place holder array, minLine is anything below 40 pixels are rejected
    averaged_lines=average_slope_intercept(frame, lines)
    line_image=display_lines(frame,lines)
    combo_image=cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow('result',combo_image)
    cv2.waitKey(1)




#plt.imshow(canny)
#plt.show()










 
