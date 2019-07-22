import cv2
from matplotlib import pyplot as plt 
from keras.layers import *
from keras.models import *



cvimg = cv2.imread('cv_test.jpg')

cv_img_rgb= cv2.cvtColor(cvimg,cv2.COLOR_BGR2RGB)

cv_img_canny = cv2.Canny(cv_img_rgb[:,:,2],100,200)
