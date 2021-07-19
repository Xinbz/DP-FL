from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from base_model import BaseModel
# from keras.applications.mobilenet import DepthwiseConv2D
import os
import imageio
import numpy as np
import scipy.misc
import argparse
import tensorflow as tf
import keras.backend as K
import keras
# from tensorflow_core.examples.tutorials.mnist import input_data
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Reshape, Add, Lambda, Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPool2D, MaxPooling2D, DepthwiseConv2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from train import train

ALPHA = 1
MODEL_NAME = f'MobileNet' # This should be modified when the model name changes.




def Mobilenet():
    '''
    Builds MobileNet.
    - MobileNets (https://arxiv.org/abs/1704.04861)
      => Depthwise Separable convolution
      => Width multiplier
    - Implementation in Keras
      => https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
    - How Depthwise conv2D works
      => https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d

    Returns:
        MobileNet model
    '''
    alpha = ALPHA # 0 < alpha <= 1
    x = Input(shape = (28, 28, 1))
    y = ZeroPadding2D(padding = (2, 2))(x) # matching the image size of CIFAR-10

    # some layers have different strides from the papers considering the size of mnist
    y = Conv2D(int(32 * alpha), (3, 3), padding = 'same')(y) # strides = (2, 2) in the paper
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = depthwise_sep_conv(y, 64, alpha) # spatial size: 32 x 32
    y = depthwise_sep_conv(y, 128, alpha, strides = (2, 2)) # spatial size: 32 x 32
    y = depthwise_sep_conv(y, 128, alpha) # spatial size: 16 x 16
    y = depthwise_sep_conv(y, 256, alpha, strides = (2, 2)) # spatial size: 8 x 8
    y = depthwise_sep_conv(y, 256, alpha) # spatial size: 8 x 8
    y = depthwise_sep_conv(y, 512, alpha, strides = (2, 2)) # spatial size: 4 x 4
    for _ in range(5):
        y = depthwise_sep_conv(y, 512, alpha) # spatial size: 4 x 4
    y = depthwise_sep_conv(y, 1024, alpha, strides = (2, 2)) # spatial size: 2 x 2
    y = depthwise_sep_conv(y, 1024, alpha) # strides = (2, 2) in the paper
    y = GlobalAveragePooling2D()(y)
    y = Dense(units = 10)(y)
    y = Activation('softmax')(y)
    model = Model(x, y)

    return model

def depthwise_sep_conv(x, filters, alpha, strides = (1, 1)):
    '''
    Creates a depthwise separable convolution block

    Args:
        x - input
        filters - the number of output filters
        alpha - width multiplier
        strides - the stride length of the convolution

    Returns:
        A depthwise separable convolution block
    '''
    y = DepthwiseConv2D((3, 3), padding = 'same', strides = strides)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(int(filters * alpha), (1, 1), padding = 'same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y



# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='NUMBER',
                    help='batch size(default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=50, metavar='NUMBER',
                    help='epochs(default: 200)')
parser.add_argument('-d', '--dataset', type=str, default="mnist", metavar='STRING',
                    help='dataset. (default: cifar10)')

args = parser.parse_args()

num_classes = 10
img_rows, img_cols = 28, 28
img_channels = 3
batch_size = args.batch_size
epochs = args.epochs
iterations = 11188 // batch_size + 1
weight_decay = 1e-4


if __name__ == '__main__':

    print("========================================")
    # print("MODEL: Residual Network ({:2d} layers)".format(6 * stack_n + 2))
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("DATASET: {:}".format(args.dataset))

    print("== LOADING DATA... ==")
    # load data
    num_classes = 10
    # if args.dataset == "cifar100":
    #     num_classes = 100
    #     (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    # else:
    #     (x_train, y_train), (x_test, y_test) = load_data()
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    public_data = np.load('/home/xbw/DP-FL-GAN/data/mnist/public_data_4000.npy')
    public_data = public_data.reshape((public_data.shape[0], 28,28,1))
    # mnist = input_data.read_data_sets('./',reshape=False)
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))/255.
    y_test = keras.utils.to_categorical(y_test,10)


    # (x_train1, y_train1), (x_test, y_test) = load_data()
    # soft_label = np.load('./data/cifar/1/soft_label_res-8.npy')
    soft_label = []
    for count in range(100):

        train = np.load('/home/xbw/DP-FL-GAN/data/mnist/1/data_%d.npy'%count)
        x_train = train[:,:784]
        x_train = x_train.reshape((x_train.shape[0],28,28,1))
        y_train = train[:,784:]
        input_shape = x_train.shape[1:]
        # x_train = x_train.astype('float32') / 255
        # public_data = public_data.astype('float32') / 255
        # if subtract_pixel_mean:
        #     x_train_mean = np.mean(x_train, axis=0)
        #     x_train -= x_train_mean
        #     public_data -= x_train_mean

        # y_train = keras.utils.to_categorical(y_train, num_classes)

        # build network
        model = Mobilenet()
        # set optimizer
        # optimizer = optimizers.RMSprop(lr=0.01)
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.summary()

        # save_dir = './models/cifar-Res-8/10/client_%d'%count
        save_dir = './models/mnist/mobilenet/1/client_%d' % count
        model_name = 'best_model.h5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True)

        # set callback
        cbks = [TensorBoard(log_dir='./Mobilenet/client_{:d}/'.format(count), histogram_freq=0),checkpoint]

        # dump checkpoint if you need.(add it to cbks)
        # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

        # set data augmentation
        print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
        train_datagen = ImageDataGenerator(rotation_range=15,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.2,
                                           zoom_range=0.1)

        # model.fit_generator(train_datagen.flow(x_train, y_train, batch_size = batch_size),
        #                      steps_per_epoch=11188 // batch_size,
        #                      epochs=epochs,
        #                      callbacks=cbks,
        #                      validation_data=(x_test, y_test))
        model.fit(x_train, y_train, batch_size = batch_size,
                             steps_per_epoch=11188 // batch_size,
                             epochs=epochs,
                             callbacks=cbks,
                             validation_data=(x_test, y_test))

        best_net = Mobilenet()
        best_net = load_model(filepath)
        soft_label.append(best_net.predict(public_data))
    np.save('./data/mnist/1/soft_label_mobile.npy',soft_label)