import time
import os, os.path
import random
import cv2
import glob
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


import pandas as pd
import numpy as np


#DIR = "./Data_example_ph2"
#codes  = os.listdir(DIR)
#codes.pop(0)
#codes.sort()

def load_imgs():
    category_dir = os.listdir(DIR)
    stats=[]
    result_imgs = []
    result_labels = []
    for thing in category_dir:
        if thing!='.DS_Store':
            label= thing
            path = os.path.join(DIR,thing)
            file_names = os.listdir(path)
            for file in file_names:
                result_labels.append(label)
                image = cv2.imread(os.path.join(path,file))
                image = cv2.resize(image, (224,224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.asarray(image)
                image =image/255
                result_imgs.append(image)
    result_imgs = np.asarray(result_imgs)
    result_labels = np.asarray(result_labels)
    return result_imgs,result_labels

#X_train,X_lables = load_imgs()

#vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))


def covnet_transform(covnet_model, raw_images):

    # Pass our training data through the network
    pred = covnet_model.predict(raw_images)

    # Flatten the array
    flat = pred.reshape(raw_images.shape[0], -1)
    
    return flat

def create_train_kmeans(data, number_of_clusters):
    # n_jobs is set to -1 to use all available CPU cores. This makes a big difference on an 8-core CPU
    # especially when the data size gets much bigger. #perfMatters
    
    k = KMeans(n_clusters=number_of_clusters, n_jobs=-1, random_state=728)
    # Let's do some timings to see how long it takes to train.
    start = time.time()

    # Train it up
    k.fit(data)

    # Stop the timing 
    end = time.time()

    # And see how long that took
    print("Training took {} seconds".format(end-start))
    
    return k

#vgg16_output = covnet_transform(vgg16_model, X_train)

#K_vgg16 = create_train_kmeans(vgg16_output)
#k_vgg16_pred = K_vgg16.predict(vgg16_output)

def cluster_label_count(clusters, labels):
    
    count = {}
    
    # Get unique clusters and labels
    unique_clusters = list(set(clusters))
    unique_labels = list(set(labels))
    
    # Create counter for each cluster/label combination and set it to 0
    for cluster in unique_clusters:
        count[cluster] = {}
        
        for label in unique_labels:
            count[cluster][label] = 0
    
    # Let's count
    for i in range(len(clusters)):
        count[clusters[i]][labels[i]] +=1
    
    cluster_df = pd.DataFrame(count)
    
    return cluster_df


#vgg16_pred_codes = [codes[x] for x in k_vgg16_pred]

from sklearn.metrics import accuracy_score, f1_score

def print_scores(true, pred):
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro")
    return "\n\tF1 Score: {0:0.8f}   |   Accuracy: {0:0.8f}".format(f1,acc)

#print("KMeans VGG16:", print_scores(X_lables, vgg16_pred_codes))



