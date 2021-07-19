import os
import imageio
import numpy as np
import scipy.misc
import tensorflow as tf
import keras.backend as K
import random
import keras
import copy
from tensorflow_core.examples.tutorials.mnist import input_data
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Reshape
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPool2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist


eps = 1
init_model = load_model('./models/mnist/1_1/client_5/init_model.h5')
W = init_model.get_weights()
W1 = copy.deepcopy(W)
W1 = np.array(W)-np.array(W1)
# sim = 0.
for i in range(100):
    model = load_model('./models/mnist/1_1/client_%d/best_model.h5'%i)
    W = model.get_weights()
    for j in range(len(W)):
        # sim += np.sum(np.abs(W[j]))
        W1[j] += 0.01*(np.array(W[j]))#+np.random.normal(0.0, 6.142795/eps, W[j].shape))
init_model.set_weights(W1)
init_model.save('./models/mnist/1_1/client_5/1_round.h5')
soft_label = []
valid_acc = []
num_client = 100
updates = []
u1 = np.load('./data/mnist/10_1/update_1_4.npy')
u2 = np.load('./data/mnist/10_1/update_2_4.npy')
u3 = np.load('./data/mnist/10_1/update_3_4.npy')
u4 = np.load('./data/mnist/10_1/update_4_4.npy')
clip1 = np.load('./data/mnist/10_1/sum_1.npy')
clip2 = np.load('./data/mnist/10_1/sum_2.npy')
clip3 = np.load('./data/mnist/10_1/sum_3.npy')
clip4 = np.load('./data/mnist/10_1/sum_4.npy')
clip = list(clip1) + list(clip2) + list(clip3) + list(clip4)
clip = np.array(clip)
clip_bound = np.median(clip)
eps = 6
sum1 = copy.deepcopy(u1[0])
sum1 = sum1 - u1[0]
for j in range(100):
    if j//25 == 0:
        u = u1
    elif j//25 == 1:
        u = u2
    elif j//25 == 2:
        u = u3
    elif j//25 == 3:
        u = u4
    for k in range(len(u)):
        for i in range(len(u[k])):
            u[k][i] = u[k][i]#/np.max([1,clip[j]/clip_bound]) #+ np.random.normal(0.0, 6.142795/eps, u[k][i].shape)
        sum1 += u[k]
sum1 += sum1/100
model = load_model('./models/mnist/10_1/client_5/init_model.h5')
W = model.get_weights()
W += sum1
model.set_weights(W)
model.save('./models/mnist/10_1/client_5/1_round.h5')


# model = load_model('./models/mnist/0.1_1/client_5/init_model.h5')
# W = model.get_weights()
# W += u1
# model.load_weights(W)

# np.save('./data/mnist/0.1_1/avg_update.npy',u1)
# for i in range(num_client):
#     model = classfier()
#     model = load_model('./models/mnist/0.1_1/client_%d/best_model.h5'%i)
#     model_init = load_model('./models/mnist/0.1_1/client_%d/init_model.h5'%i)
#
#     W = model.get_weights()
#     W_t = model_init.get_weights()
#     count = 0.
#     for i in range(len(W)):
#         count += np.sum(np.abs(W_t[i]-W[i]))
#     updates.append(count)
# np.median(new)
#
# # best_net = Mobilenet()
# # best_net = load_model(filepath)
# # soft_label.append(best_net.predict(public_data))
# # model.save('./models/mnist/10/client_%d.h5'%(i))
# np.save('./data/mnist/retrain_0.1/100_client_1st_soft_label.npy',soft_label)
# np.save('./data/mnist/retrain_0.1/100_client_1st_valid_acc.npy',valid_acc)
# np.save('./data/mnist/retrain_0.1/40_client_2nd_soft_label.npy',soft_label)
# np.save('./data/mnist/retrain_0.1/40_client_2nd_valid_acc.npy',valid_acc)