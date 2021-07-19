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
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist


av = []
account = []
avg = []
num_client = 400
num_sample = 2000
smooth = 40.
eps = 2.5
ratio = 0

b = [0.6,0.8]
public_data = np.load('./data/cifar/public_data_%d.npy'%num_sample)
public_labels = np.load('./data/cifar/public_labels_%d.npy'%num_sample)
public_labels = np.argmax(public_labels,axis=1)
soft_label_pro = np.load('./data/cifar/0.1/soft_label.npy')
M_soft_label_pro = np.load('./data/cifar/0.1/Res_8_soft_label.npy')
soft_label_pro = list(soft_label_pro)
M_soft_label_pro= list(M_soft_label_pro)

for i in range(num_client-100):
    r = random.randint(0,99)
    soft_label_pro.append(soft_label_pro[r])
for i in range(num_client - 100):
    r = random.randint(0, 99)
    soft_label_pro.append(soft_label_pro[r])
    M_soft_label_pro.append(M_soft_label_pro[r])

for p in range(num_client):
    for q in range(2000):
        soft_label_pro[p][q] += np.random.laplace(0,2./eps,10)
        M_soft_label_pro[p][q] += np.random.laplace(0,2./eps, 10)

soft_label_1 = []
for j in range(num_client):
    if j < ratio*num_client:
        soft_label_1.append(soft_label_pro[j])
    else:
        soft_label_1.append(M_soft_label_pro[j])


public_labels = public_labels[:num_sample]
# for p in range(num_client):
#     for q in range(num_sample):
#         soft_label_pro[p][q] += np.random.laplace(0,2/eps,10)



c = np.load('./data/cifar/clustering_%d.npy'%num_sample)
a = {}
for i in range(10):
    a[i] = []
for i in range(num_sample):
    a[c[i]].append(i)

my_list = [elem for elem in a.values()]

for w in range(20):
    soft_label = copy.deepcopy(soft_label_pro[:num_client])
    # soft_label = copy.deepcopy(soft_label_1[:num_client])
    for aa in range(num_client):
        for bb in range(len(my_list)):
            if len(my_list[bb]) != 0:
                random.shuffle(my_list[bb])
                # temp1 = my_list[bb][0]
                jcq = []
                sumf = np.zeros([1, 10])
                if (smooth >= len(my_list[bb])):
                    avgf = np.sum(soft_label[aa][my_list[bb]], axis=0) / len(my_list[bb])
                    # avgf += np.random.laplace(0, 0.89, 10)
                    for cc in my_list[bb]:
                        soft_label[aa][cc] = avgf
                    avgf = np.zeros([1, 10])
                else:
                    for cc in range(len(my_list[bb])):
                        if (cc % int(smooth) == 0) and (cc != 0):
                            avgf = sumf / smooth
                            # avgf += np.random.laplace(0,0.89,10)
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

    soft_labels = np.zeros([num_sample,10])
    for j in range(num_sample):
        count = 0.
        num = np.zeros(soft_labels[0].shape)
        for n in range(num_client):
            p = np.sort(soft_label[n][j])
            if p[-1] - p[-2] > b[0]:
                count += 1
                num += soft_label[n][j]
        if count != 0:
            soft_labels[j] = num/count

    # soft_label = np.array(soft_label)
    # soft_label = soft_label.transpose((1, 0, 2))
    # for j in range(num_sample):
    #     avgf = np.sum(soft_label[j], axis=0) / num_client
    #     soft_labels[j] = avgf


    index = np.where(np.amax(soft_labels,axis=1)<b[1])
    final_l_90 = np.delete(soft_labels,index,0)
    final_l_90 = np.argmax(final_l_90,axis=1)
    real_l_90 = np.delete(public_labels,index,0)
    diff = real_l_90-final_l_90
    x = np.where(diff == 0)
    av.append(x[0].size)
    account.append(diff.size)
    avg.append(x[0].size / diff.size)
    print(avg[w])
    # print('1')
    # re_data = np.hstack((public_data, soft_labels))
    # re_data = np.delete(re_data, index, 0)
    # np.save('./data/cifar/0.1/eps_1.npy',re_data)
print(np.average(av), np.average(account), np.average(avg))

print(0)
    # print(avg[w])