import os
import imageio
import numpy as np
import scipy.misc
import tensorflow as tf
import keras.backend as K
import keras
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

# model = load_model('./models/model_1.h5')
# model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
# public_prediction = np.load('./data/2_round_public_data.npy')

# test_data = np.load('./data/test_data.npy')
pro_public_data = np.load('./data/public_data.npy')
public_data = pro_public_data[:,:784]
public_data = np.reshape(public_data,(public_data.shape[0],28,28,1))
mnist = input_data.pro_read_data_set('./',reshape=False)
x_test = mnist.validation.images
valid_labels = keras.utils.to_categorical(mnist.validation.labels,10)
y_test = valid_labels
# soft_labels = np.zeros([2000,10])



# mnist = input_data.read_data_sets('./',reshape = False)
# model = classfier()
# model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
# labels = keras.utils.to_categorical(mnist.train.labels,10)
# model.fit(mnist.train.images, labels, batch_size=64, epochs=100, shuffle=True)
# model.save('./models/mnist/model.h5')








soft_label = []
valid_acc = []
for i in range(0,100):
    train = np.load('./data/mnist/1/data_%d.npy'%(i))
    # train = np.vstack((train,public_prediction))
    # train = public_prediction
    train_img = train[:,:784]
    train_img = np.reshape(train_img,(train_img.shape[0],28,28,1))
    # mnist = input_data.read_data_sets(train_img,train[:,784:])
    model = classfier()
    save_dir = './models/mnist/1_1/client_%d' % i
    model_name = 'best_model.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
    cbks = [TensorBoard(log_dir='./vgg/client_{:d}/'.format(i), histogram_freq=0),
            checkpoint]
    # model = load_model('./model/client_%d.h5'%(i+1))
    # model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    model.load_weights('./models/mnist/1_1/client_5/init_model.h5')
    # model.summary()
    # model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
    model.fit(train_img, train[:,784:], batch_size=64, epochs=20, shuffle=True,callbacks=cbks,validation_data=(x_test, y_test))
    print('the %d model trained'%i)


# np.save('./data/mnist/10/soft_label.npy',soft_label)
# np.save('./data/mnist/10/valid_acc.npy',valid_acc)