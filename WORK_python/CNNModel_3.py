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


def conv_block_type1(x,num_filters, kernel_size):
    x = Conv1D(num_filters, kernel_size, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Conv1D(num_filters, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling1D()(x)

    return x


def conv_block_type2(x,num_filters, kernel_size):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Conv1D(num_filters, kernel_size, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Conv1D(num_filters, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling1D()(x)

    return x


def build_model(time_step, num_sensors, num_classes):
    input_shape = (time_step, num_sensors)

    #CNNModel
    inpt = Input(input_shape)

    x = Conv1D(32, 16, activation='relu', input_shape=input_shape)(inpt)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = conv_block_type1(x, 32, 16)

    x = conv_block_type2(x, 32, 16)
    x = conv_block_type2(x, 64, 8)
    x = conv_block_type2(x, 64, 8)
    x = conv_block_type2(x, 128, 4)
    x = conv_block_type2(x, 128, 4)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True, input_shape=(None, 1)))(x)
    x = GlobalMaxPooling1D()(x)

    x = Dense(32, activation='relu', input_shape=(None, 128))(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(input=inpt, output=x)

    return(model)