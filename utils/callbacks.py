# -*- coding: utf-8 -*-
'''
Author: wayne
LastEditors: wayne
email: linzhihui@szarobots.com
Date: 2022-05-25 16:50:20
LastEditTime: 2022-05-26 16:04:35
Description: 
'''

import keras

class MyCbk(keras.callbacks.Callback):

    def __init__(self, model):
         self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('Logs/model_at_epoch_%d.h5' % (epoch+1))