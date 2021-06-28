import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras import backend as keras


def unet(pretrained_weights = None,input_size = (256,256,3)):
    input_img = Input(input_size)
    num_filter = 64
    kernel_size = 7
    strides = 1
    activation = 'tanh'

    x = Conv2D(num_filter, kernel_size, 1, activation=activation, padding='same', name='enc1')(input_img)
    # x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    # x = MaxPooling2D((2, 2))(x)

    x = Conv2D(num_filter, kernel_size, strides, activation=activation, padding='same', name='enc2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(num_filter, kernel_size, strides, activation=activation, padding='same', name='enc3')(x)
    x = InstanceNormalization()(x)
    # x = MaxPooling2D((2, 2))(x)

    encoded = Conv2D(3, 1, 1, activation=activation, padding='valid', name='encoder_output')(x)

    # x = UpSampling2D((2, 2), interpolation='bilinear')(encoded)
    x = Conv2DTranspose(num_filter, kernel_size, 1, activation=activation, padding='same', name='dec1')(encoded)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = Conv2DTranspose(num_filter, kernel_size, 1, activation=activation, padding='same', name='dec2')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(num_filter, kernel_size, 1, activation=activation, padding='same', name='dec3')(x)
    x = BatchNormalization()(x)

    decoded = Conv2D(3, (1, 1), activation='sigmoid', name='output')(x)

    model = Model(inputs = input_img, outputs = decoded)

    model.compile(optimizer = Adam(), loss = 'mean_squared_error', metrics = ['mae'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


