import numpy as np
import keras
import os, pickle
from tensorflow.examples.tutorials.mnist import input_data
from load_data import load_data


def print_split(idcs, labels):
  n_labels = np.max(labels) + 1
  print("Data split:")
  splits = []
  for i, idccs in enumerate(idcs):
    split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
    splits += [split]
    if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
      print(" - Client {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
    elif i==len(idcs)-10:
      print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)

  print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
  print()
def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    return x
# def split_dirichlet(labels, n_clients, n_data, alpha, double_stochstic=True, seed=0):
#     '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

(x_train1, y_train1), (x_test, y_test) = load_data()
# images = np.load('./data/cifar_train_images.npy')
# labels = np.load('./data/cifar_train_labels.npy')
images = x_train1
labels1 = y_train1
labels = [labels1[x][0] for x in range(len(labels1))]
labels = np.array(labels)
n_clients = 4
alpha = 0.1
seed = 0
double_stochstic=True
np.random.seed(seed)

# if isinstance(labels, torch.Tensor):
#   labels = labels.numpy()
for count in range(25):
    n_classes = np.max(labels)+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels)==y).flatten()
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    for i in range(len(client_idcs)):
        pro_images = images[client_idcs[i]]
        post_images = np.reshape(pro_images,(pro_images.shape[0],-1))
        pro_labels = labels[client_idcs[i]]
        post_labels = keras.utils.to_categorical(pro_labels, 10)
        dataset = np.hstack((post_images,post_labels))
        np.save('./data/cifar/0.1/data_%d.npy'%(i+4*count), dataset)


# print_split(client_idcs, labels)