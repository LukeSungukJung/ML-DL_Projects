import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import PIL
import zipfile
import glob
from PIL import ImageOps, ImageFilter, ImageDraw
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import *
from keras import *

data_dir = os.listdir('sample_data')

train_data_path =  os.path.join('sample_data','train')
df_train = pd.read_csv(os.path.join('sample_data','train.csv'))

def crop_boxing_img(img_name,margin=10):
    if img_name.split('_')[0] ==  "train":
        path =  train_data_path
        data =  df_train
    #elif img_name.split('_')[0] =="test":
    img = PIL.Image.open(os.path.join(path, img_name))
    pos = data.loc[data["img_file"] == img_name,['bbox_x1','bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)
    width,height = img.size
    x1 = max(0, pos[0] - margin);
    y1 = max(0, pos[1] - margin)
    x2 = min(pos[2] + margin, width)
    y2 = min(pos[3] + margin, height)
    
    return img.crop((x1,y1,x2,y2))

for i, row in df_train.iterrows():
    cropped = crop_boxing_img(row['img_file'])
    cropped.save('cropped/'+row['img_file'])

preprocessed_imgs = glob.glob('cropped*.jpg')
lables =  list(df_train['class'])
datagen = image.ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,zoom_range=0.2,
                             horizontal_flip=True,vertical_flip=False,fill_mode='nearest')
cropped_train_path = 'cropped'


train_generator = datagen.flow_from_directory(
    cropped_train_path,
    target_size=(224, 224),
    batch_size=3,
    class_mode='categorical',
    seed=2019,
    color_mode='rgb'
)

model =  models.Sequential()
conv_base =VGG16(input_shape=(224,224,3))
conv_model_layers = conv_base.layers[:-3]

for conv_model in conv_model_layers:
    model.add(conv_model)

model.add(layers.Dense(100,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8,activation='softmax'))
model.compile(optimizer=optimizers.Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['acc'])
history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=5)




