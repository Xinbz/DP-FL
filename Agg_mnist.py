import os
import imageio
import numpy as np
import scipy.misc
import tensorflow as tf
import keras.backend as K
import keras
import random
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

def classfier():

    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=(28,28,1), padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    img = Input(shape=(28, 28, 1))
    validity = model(img)
    model = Model(inputs=img, outputs=validity)
    return model
public_data = np.load('./data/mnist/public_data.npy')
public_labels = np.load('./data/mnist/public_labels.npy')
# soft_label_pro = np.load('./data/mnist/retrain_0.1/40_client_soft_label.npy')
soft_label_pro = np.load('./data/mnist/0.1/soft_label.npy')
M_soft_label_pro = np.load('./data/mnist/0.1/Mobile_soft_label.npy')

c = np.load('./data/mnist/10-clustering_2000.npy')
# c = public_labels
a = {}
for i in range(10):
    a[i] = []
for i in range(2000):
    a[c[i]].append(i)

av = []
account = []
avg = []
soft_label_pro = list(soft_label_pro)
M_soft_label_pro = list(M_soft_label_pro)
num_client = 400
eps = 0.8
smooth = 40.
b = [0.5,0.8]
ratio = 0.0

soft_label_1 = []

for i in range(num_client - 100):
    r = random.randint(0, 99)
    soft_label_pro.append(soft_label_pro[r])
    M_soft_label_pro.append(M_soft_label_pro[r])

for p in range(num_client):
    for q in range(2000):
        soft_label_pro[p][q] += np.random.laplace(0,2./eps,10)
        # M_soft_label_pro[p][q] += np.random.laplace(0, 2./eps, 10)


for j in range(num_client):
    if j < ratio*num_client:
        soft_label_1.append(soft_label_pro[j])
    else:
        soft_label_1.append(M_soft_label_pro[j])


# for i in range(num_client-100):
#     # r = random.randint(0,99)
#     r = i%100
#     soft_label_pro.append(soft_label_pro[r])
for k in range(20):
    soft_label = copy.deepcopy(soft_label_pro[:num_client])
    # soft_label = copy.deepcopy(soft_label_1[:num_client])

    my_list = [elem for elem in a.values()]
    for aa in range(num_client):
        for bb in range(len(my_list)):
            if len(my_list[bb]) != 0:
                random.shuffle(my_list[bb])
                # temp1 = my_list[bb][0]
                jcq = []
                sumf = np.zeros([1, 10])
                if (smooth >= len(my_list[bb])):
                    avgf = np.sum(soft_label[aa][my_list[bb]], axis=0) / len(my_list[bb])
                    # avgf += np.random.laplace(0, 2, 10)
                    for cc in my_list[bb]:
                        soft_label[aa][cc] = avgf
                    avgf = np.zeros([1, 10])
                else:
                    for cc in range(len(my_list[bb])):
                        if (cc % int(smooth) == 0) and (cc != 0):
                            avgf = sumf / smooth
                            # avgf += np.random.laplace(0,2,10)
                            for dd in jcq:
                                soft_label[aa][dd] = avgf
                            jcq.clear()
                            sumf = np.zeros([1, 10])
                            avgf = np.zeros([1, 10])
                            jcq.append(my_list[bb][cc])
                            sumf += soft_label[aa][my_list[bb][cc]]
                        else:
                            jcq.append(my_list[bb][cc])
                            sumf += soft_label[aa][my_list[bb][cc]]

    soft_labels = np.zeros([2000,10])
    for j in range(2000):
        count = 0.
        num = np.zeros(soft_labels[0].shape)
        for n in range(num_client):
            p = np.sort(soft_label[n][j])
            if p[-1] - p[-2] > b[0]:
                count += 1
                num += soft_label[n][j]
        if count != 0:
            soft_labels[j] = num/count
    index = np.where(np.amax(soft_labels,axis=1)<b[1])
    final_l_90 = np.delete(soft_labels,index,0)
    final_l_90_1 = np.argmax(final_l_90,axis=1)
    real_l_90 = np.delete(public_labels,index,0)
    diff = real_l_90-final_l_90_1
    x = np.where(diff==0)
    av.append(x[0].size)
    account.append(diff.size)
    avg.append(x[0].size/diff.size)
    print(avg[k])
print(np.average(av), np.average(account), np.average(avg))

    # re_data = np.hstack((public_data, soft_labels))
    # re_data = np.delete(re_data, index, 0)
    # np.save('./data/mnist/0.1/2_round_data.npy',re_data)
ax = np.array(account)
bx = np.array(avg)
cx = np.array(av)
print(np.average(cx),np.average(ax),np.average(bx))