# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import argparse
import time
import pandas as pd
import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from sklearn.cluster.k_means_ import KMeans


def convet_transform_gen(covnet_model, generator):

    # Pass our training data through the network
    pred = covnet_model.predict_generator(generator,steps=len(generator),verbose=1)
    pred= np.asarray(pred)
    # Flatten the array
    flat = pred.reshape(pred.shape[0], -1)
    
    return flat

def convet_transform(covnet_model, raw_images):

    # Pass our training data through the network
    pred = covnet_model.predict(raw_images)

    # Flatten the array
    flat = pred.reshape(raw_images.shape[0], -1)
    
    return flat


def load_imgs(dir_):
    category_dir = os.listdir(dir_)
    stats=[]
    result_imgs = []
    result_labels = []
    for thing in category_dir:
        if thing!='.DS_Store':
            label= thing
            path = os.path.join(dir_,thing)
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

def load_codes(path):
    DIR = path
    category_dir = os.listdir(DIR)
    if category_dir[0]=='.DS_Store':
        category_dir.pop(0)
    return category_dir

from sklearn.metrics import accuracy_score, f1_score

def print_scores(true, pred):
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro")
    return "\n\tF1 Score: {0:0.8f}   |   Accuracy: {0:0.8f}".format(f1,acc)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, _):
        test_path = DATASET_PATH + '/test/test_data'

        db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]

        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        queries, query_vecs, references, reference_vecs = get_feature(model, queries, db)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:1000]

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# data preprocess
def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'
    codes = load_codes(test_path)
    print(codes)
    
    def create_train_kmeans(data, number_of_clusters = len(codes)):
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

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=-3).output)
    
    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32')
    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['query'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    
    #query_x,query_label =  load_imgs(test_path)
    #query_vecs = convet_transform(intermediate_layer_model,query_x) 
    query_vecs = convet_transform_gen(intermediate_layer_model,query_generator)
    K_query = create_train_kmeans(query_vecs)
    query_vecs =K_query.cluster_centers_
    
    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['reference'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    #reference_x,reference_label =  load_imgs(test_path)
    reference_vecs = convet_transform_gen(intermediate_layer_model,reference_generator)
    #reference_vecs = convet_transform(intermediate_layer_model,reference_x)
    K_reference = create_train_kmeans(reference_vecs) 
    reference_vecs = K_reference.cluster_centers_

    return queries, query_vecs, db, reference_vecs


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=5)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = 30
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape

    """ Model """
    model = InceptionV3(weights=None,classes=num_classes)
    model.summary()

    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate RMSprop optimizer """
        opt = keras.optimizers.rmsprop(lr=0.00045, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        print('dataset path', DATASET_PATH)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )

        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
            nsml.save(epoch)
        print('Total training time : %.1f' % (time.time() - t0))