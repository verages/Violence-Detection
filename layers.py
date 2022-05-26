# -*- coding: utf-8 -*-
'''
Author: wayne
LastEditors: wayne
email: linzhihui@szarobots.com
Date: 2022-05-25 15:10:41
LastEditTime: 2022-05-26 16:29:25
Description: FGN layers
'''
from keras.models import Model
from keras.layers import Input,Dense, Flatten, Conv3D, MaxPooling3D, Dropout, Multiply
from keras.layers.core import Lambda

# extract the rgb images 
def get_rgb(input_x):
    rgb = input_x[...,:3]
    return rgb

# extract the optical flows
def get_opt(input_x):
    opt= input_x[...,3:5]
    return opt

# RGB channel
def rgb_block(inputs):
    rgb = Lambda(get_rgb,output_shape=None)(inputs)
    rgb = Conv3D(
        16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(
        16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(
        32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(
        32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)
    return rgb

# Optical Flow channel
def opt_block(inputs):
    opt = Lambda(get_opt,output_shape=None)(inputs)
    opt = Conv3D(
        16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = Conv3D(
        16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)

    opt = Conv3D(
        16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = Conv3D(
        16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)

    opt = Conv3D(
        32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = Conv3D(
        32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)

    opt = Conv3D(
        32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(opt)
    opt = Conv3D(
        32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)
    return opt


# Fusion and Pooling
def fusion_block(rgb,opt):
    x = Multiply()([rgb,opt])
    x = MaxPooling3D(pool_size=(8,1,1))(x)

    # Merging Block
    x = Conv3D(
        64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(
        64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(
        64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(
        64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(
        128, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(
        128, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,3,3))(x)
    return x

# FGN network
def FGN(inputs):
    rgb = rgb_block(inputs)
    opt = opt_block(inputs)
    x = fusion_block(rgb,opt)
    x = Flatten()(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    pred = Dense(2, activation='softmax')(x)
    return Model(inputs=inputs, outputs=pred)

# # Build the model
# input_x = Input(shape=(64,224,224,5))
# model = FGN(input_x)
# model.summary()
# out = model.get_layer('dense_2')
# print(out.output)
# print(model.input)
