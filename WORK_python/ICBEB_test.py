import csv

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

from score_py3 import score

path_test = "E:\BIG_WORK\PhysioNet2017\WORK_PhysioNet2017\WORK_DATA\DATA\DATA_811_folds_1\DATA_test/"
path_test_ref = "E:\BIG_WORK\PhysioNet2017\WORK_PhysioNet2017\WORK_DATA\DATA\DATA_811_folds_1\DATA_test/REFERENCE.csv"
path_test_answer ="E:\BIG_WORK\PhysioNet2017\WORK_PhysioNet2017\WORK_DATA\DATA\DATA_811_folds_1/answers.csv"

#Hyper Parameters
time_step = 18000
num_sensors = 1


def normalize(v):
    return (v - v.mean(axis=1).reshape((v.shape[0],1))) / (v.max(axis=1).reshape((v.shape[0],1)) + 2e-12)


#loadmat
def get_feature(wav_file, path_data):
    mat = loadmat(path_data + wav_file)
    ECG = mat['val']
    dat = ECG

    feature = sequence.pad_sequences(dat, maxlen=time_step, dtype='float32', truncating='post')

    # 
    return (normalize(feature).transpose())  # return(normalize(feature))


#oneHot
def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[index-1] = 1
    return(hot)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = load_model("best_model.109-0.87.h5")
    pre_lists = pd.read_csv(path_test_ref)
    print(pre_lists.head())
    pre_lists = np.array(pre_lists)
    pre_datas = np.array([get_feature(item, path_test) for item in pre_lists[:,0]])

    #for Sequential model
    #pre_result = model.predict_classes(pre_datas)

    #for other model
    pre_r = model.predict(pre_datas)
    pre_result = np.argmax(model.predict(pre_datas), axis=1)

    print(pre_result.shape)
    result_label = [x+1 for x in pre_result]

    df1 = np.array([pre_lists[:,0]]).T
    df2 = np.array([result_label]).T
    df = np.hstack((df1, df2))

    header = np.array(['Recording', 'Result'])
    answer = np.vstack((header,df))

    dataframe = pd.DataFrame(answer)
    dataframe.to_csv(path_test_answer, header=False)

    print("predict finish")

    #Performance Evaluation
    score(path_test_answer, path_test_ref)