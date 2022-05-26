# -*- coding: utf-8 -*-
'''
Author: wayne
LastEditors: wayne
email: linzhihui@szarobots.com
Date: 2022-05-25 16:47:28
LastEditTime: 2022-05-26 16:06:54
Description: train.py
'''
from termcolor import colored
import argparse
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from keras.callbacks import LearningRateScheduler,CSVLogger
from keras.layers import Input
from layers import FGN
import keras.backend as K
from tensorflow.keras.optimizers import SGD,Adam
from utils.callbacks import MyCbk
from Preprocess.dataset import DataGenerator

input_x = Input(shape=(64,224,224,5))
parallel_model = None

def scheduler(epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(parallel_model.optimizer.lr)
        K.set_value(parallel_model.optimizer.lr, lr * 0.7)
    return K.get_value(parallel_model.optimizer.lr)

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu', help=' cpu or gpu ')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--dataset_name', type=str, default='npy')
    parser.add_argument('--num_gpus', type=int, default=2)
    opt = parser.parse_args()
    return opt

def main(opt):
    global parallel_model
    # - set essential params
    num_epochs  = opt.num_epochs
    num_workers = opt.num_workers
    batch_size  = opt.batch_size
    dataset_name = opt.dataset_name
    num_gpus    = opt.num_gpus
    device      = opt.device


    # - build model
    model = FGN(input_x)
    model.summary()

    # - set run mode
    if device == 'gpu':
        parallel_model = multi_gpu_model(model, gpus=num_gpus)
    else:
        parallel_model = model
    

    # - Model Compiling
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    parallel_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = LearningRateScheduler(scheduler)
    check_point = MyCbk(parallel_model)

    filename = 'Logs/ours_log.csv'
    csv_logger = CSVLogger(filename, separator=',', append=True)

    callbacks_list = [check_point, csv_logger, reduce_lr]

    dataset = dataset_name

    train_generator = DataGenerator(directory='./Datasets/{}/train'.format(dataset), 
                                    batch_size=batch_size, 
                                    data_augmentation=True)

    val_generator = DataGenerator(directory='./Datasets/{}/val'.format(dataset),
                                batch_size=batch_size, 
                                data_augmentation=False)
    
    hist = parallel_model.fit_generator(
    generator=train_generator, 
    validation_data=val_generator,
    callbacks=callbacks_list,
    verbose=1, 
    epochs=num_epochs,
    workers=num_workers ,
    max_queue_size=4,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator))




    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)