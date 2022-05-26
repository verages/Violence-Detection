# -*- coding: utf-8 -*-
'''
Author: wayne
LastEditors: wayne
email: linzhihui@szarobots.com
Date: 2022-05-26 10:19:10
LastEditTime: 2022-05-26 16:29:13
Description: infer video
'''

import os
import numpy as np
import keras
from Preprocess.Video2Numpy import Video2Npy,write_video

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = keras.models.load_model('./Models/keras_model.h5')
model.summary()  


if __name__ == "__main__":

    video_path = './Video/t1.mp4'
    status = []
    data = Video2Npy(file_path=video_path, resize=(224,224))
    data = np.uint8(data)
    for j in range(len(data) // 64):
        data_test = np.expand_dims(data[64*j:64*(j+1)],axis=0)
        tr = model.predict(data_test)
        if tr[0][0]>tr[0][1]:
            for i in range (64):
                status.append(True)
        else:
            for i in range (64):
                status.append(False)
    write_video(video_path,status)

