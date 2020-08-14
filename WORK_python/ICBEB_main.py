import keras
from scipy.io import loadmat
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import math
import os
from keras.layers import *
from keras.models import *
from keras.objectives import *
from keras.preprocessing import sequence
from keras.utils import plot_model

from CNNModel_3 import build_model

#solve plot_model problem
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# Data Path
path_train = "E:\BIG_WORK\PhysioNet2017\WORK_PhysioNet2017\WORK_DATA\DATA\DATA_811_folds_1\DATA_train/"
path_train_ref = "E:\BIG_WORK\PhysioNet2017\WORK_PhysioNet2017\WORK_DATA\DATA\DATA_811_folds_1\DATA_train/REFERENCE.csv"
path_validation = "E:\BIG_WORK\PhysioNet2017\WORK_PhysioNet2017\WORK_DATA\DATA\DATA_811_folds_1\DATA_test/"
path_validation_ref = "E:\BIG_WORK\PhysioNet2017\WORK_PhysioNet2017\WORK_DATA\DATA\DATA_811_folds_1\DATA_test/REFERENCE.csv"

# Hyper Parameters
time_step = 18000
num_sensors = 1
num_classes = 4
Batch_size = 32
num_epoch = 300


#
def normalize(v):
    return (v - v.mean(axis=1).reshape((v.shape[0], 1))) / (v.max(axis=1).reshape((v.shape[0], 1)) + 2e-12)


# loadmat
def get_feature(wav_file, path_data):
    mat = loadmat(path_data + wav_file)
    ECG = mat['val']
    # dat = ECG[0, 0]['data']
    dat = ECG

    feature = sequence.pad_sequences(dat, maxlen=time_step, dtype='float32', truncating='post')

    # 
    return (normalize(feature).transpose())  # return(normalize(feature))


# oneHot
def convert2oneHot(index, Lens):
    hot = np.zeros((Lens,))
    hot[index - 1] = 1
    return (hot)


def xs_gen(path_ref, path_data, batch_size=Batch_size, train=True):
    img_list = pd.read_csv(path_ref)

    if train:
        img_list = np.array(img_list)
        print("Found %s train items." % len(img_list))
        print("list 1 is", img_list[0])
        steps = math.ceil(len(img_list) / batch_size)  
    else:
        img_list = np.array(img_list)
        print("Found %s test items." % len(img_list))
        print("list 1 is", img_list[0])
        steps = math.ceil(len(img_list) / batch_size)  
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size: i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            batch_x = np.array([get_feature(file, path_data) for file in batch_list[:, 0]])
            batch_y = np.array([convert2oneHot(label, 4) for label in batch_list[:, 1]])

            yield batch_x, batch_y


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # dat1 = get_feature("A1001.mat")
    # print("one data shape is",dat1.shape)
    # one data shape is (5000, 12)
    # plt.plot(dat1[:,0])
    # plt.show()

    img_list = pd.read_csv(path_train_ref)
    len_img = np.shape(img_list)
    num_train = len_img[0]

    img_list = pd.read_csv(path_validation_ref)
    len_img = np.shape(img_list)
    num_validation = len_img[0]

    train_iter = xs_gen(path_ref=path_train_ref, path_data=path_train, train=True)
    validation_iter = xs_gen(path_ref=path_validation_ref, path_data=path_validation, train=False)

    model = build_model(time_step, num_sensors, num_classes)
    print(model.summary())
    #plot_model(model, to_file='current_model.png')

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_acc:.2f}.h5',
        monitor='val_acc', save_best_only=True, verbose=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit_generator(
        generator=train_iter,
        steps_per_epoch=num_train // Batch_size,
        epochs=num_epoch,
        initial_epoch=0,
        validation_data=validation_iter,
        nb_val_samples=num_validation // Batch_size,
        callbacks=[ckpt],
    )
