import os,shutil
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import *
from keras import *

#run 1 walk 0
original_dataset_dir ='./datasets'

origin_train_run = os.path.join(original_dataset_dir,'train/run')
origin_train_walk= os.path.join(original_dataset_dir,'train/walk')

origin_test_run = os.path.join(original_dataset_dir,'test/run')
origin_test_walk = os.path.join(original_dataset_dir,'test/walk')

all_test_walk =glob.glob(os.path.join(origin_test_walk,'*.png'))
all_test_run =glob.glob(os.path.join(origin_test_run,'*.png'))
all_test = all_test_walk+ all_test_run
all_test_label =  [0]*len(all_test_walk)+[1]*len(all_test_run)

all_train_walk = glob.glob(os.path.join(origin_train_walk,'*.png'))
all_train_run = glob.glob(os.path.join(origin_train_run,'*.png'))
all_walk_labels = [1]*len(all_train_walk)
all_run_labels= [0]* len(all_train_run)


divided_folder= os.path.join(original_dataset_dir,'divided')
validataion_location = os.path.join(divided_folder,'validataion')
test_location = os.path.join(divided_folder,'test')
train_location = os.path.join(divided_folder,'train')


os.mkdir(divided_folder)
os.mkdir(validataion_location)
os.mkdir(test_location)
os.mkdir(train_location)

run_train = os.path.join(train_location,'run')
walk_train =os.path.join(train_location,'walk')
run_test= os.path.join(test_location,'run')
walk_test= os.path.join(test_location,'walk')
validation_walk = os.path.join(validataion_location,'walk')
validation_run = os.path.join(validataion_location,'run')

os.mkdir(run_train)
os.mkdir(walk_train)
os.mkdir(run_test)
os.mkdir(walk_test)
os.mkdir(validation_walk)
os.mkdir(validation_run)

all_train = all_train_walk +all_train_run
all_train_labels = all_walk_labels+ all_run_labels
all_train_index = np.array(range(len(all_train)))
np.random.shuffle(all_train_index)
all_train =[all_train[i] for i in all_train_index]
all_train_labels =[all_train_labels[i] for i in all_train_index]



validation_data = all_train[:100]
validation_data = np.asarray([cv2.imread(img) for img in validation_data])
validation_label = all_train_labels[:100]

train_data = all_train[100:]
train_data =np.asarray([cv2.imread(img) for img in train_data])
train_labels = all_train_labels[100:]

for data_piece in all_train_walk[50:]:
    shutil.copy(data_piece,walk_train)
    
for data_piece in all_train_run[50:]:
    shutil.copy(data_piece,run_train)
    
for data_piece in all_test_walk[50:]:
    shutil.copy(data_piece,walk_test)
    
for data_piece in all_test_run[50:]:
    shutil.copy(data_piece,run_test)
    
for data_piece in all_train_walk[:50]:
    shutil.copy(data_piece,validation_walk)
    
for data_piece in all_train_run[:50]:
    shutil.copy(data_piece,validation_run)
    


datagen = image.ImageDataGenerator(rescale=1./255,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,zoom_range=0.2,
                             horizontal_flip=True,fill_mode='nearest')

test_gen = image.ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(train_location,target_size=(224,224),batch_size=32,class_mode='binary')
test_generator = datagen.flow_from_directory(test_location,target_size=(224,224),batch_size=32,class_mode='binary')
validataion_generator = datagen.flow_from_directory(validataion_location,target_size=(224,224),batch_size=32,class_mode='binary')

model =  models.Sequential()
conv_base =VGG16(input_shape=(224,224,3))
conv_model_layers = conv_base.layers[:-3]

for conv_model in conv_model_layers:
    model.add(conv_model)

model.add(layers.Dense(100,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=optimizers.Adam(lr=1e-5),loss='binary_crossentropy',metrics=['acc'])
history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validataion_generator,validation_steps=50)
model.save('walk and run_v2.h5')









