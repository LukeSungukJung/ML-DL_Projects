import numpy as np
import cv2
from keras.applications.vgg19 import vgg19
from keras.layers import *
from keras.models import *

TRAIN_CROPPED_PATH = '../cropped_train'
TEST_CROPPED_PATH = '../cropped_test'
VALID_CROPPED_PATH = '../cropped_valid'

r_model = vgg19.VGG19(input_shape=(224,224),weights='imagenet',include_top=False)
g_model = vgg19.VGG19(input_shape=(224,224),weights='imagenet',include_top=False)
b_model = vgg19.VGG19(input_shape=(224,224),weights='imagenet',include_top=False)

