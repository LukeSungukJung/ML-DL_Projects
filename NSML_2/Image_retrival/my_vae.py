from keras.layers import *
from keras.models import *
from tensorflow.keras.backend import tanh, sigmoid
from keras.optimizers import *
from keras import metrics
import cv2
import numpy as np
import os

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch, latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


train_dir = './Data_example_ph2'
train_img = []
for img_folder in os.listdir(train_dir):
    if img_folder!='.DS_Store':
        folder_path = os.path.join(train_dir,img_folder)
        for img in (os.listdir(folder_path)):
            img = cv2.imread(os.path.join(folder_path,img),1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[250:750, 250:750]
            img = cv2.resize(img, input_shape[:2])
            train_img.append(img)
        
train_img = np.asarray(train_img)
train_img = train_img/255
input_img = Input(shape=(224,224, 3))  # adapt this if using `channels_first` image data format
latent_dim =100
input_layer = input_img
x = Conv2D(64, (3, 3), padding='same')(input_layer)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
    
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)
z = Lambda(sampling)([z_mean, z_log_var])


x = Dense(14*14*32)(z)
x = Reshape((14, 14, 32))(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = UpSampling2D(size=(4, 4))(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = UpSampling2D(size=(4, 4))(x)
x = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

def vae_loss(x, z_decoded):
    x = K.flatten(x)
    z_decoded = K.flatten(z_decoded)
    # Reconstruction loss
    xent_loss = metrics.binary_crossentropy(x, z_decoded)
    # KL divergence
    # mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    
    kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)


autoencoder = Model(input_img, x)
autoencoder.compile(optimizer=RMSprop(5e-5), loss=vae_loss)
autoencoder.fit(x=train_img,y=train_img,
                    batch_size=4,
                    epochs=20,
                    shuffle=True)
