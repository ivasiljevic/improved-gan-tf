import numpy as np
import os
import scipy.io as sio

train = sio.loadmat("/share/data/vision-greg/svhn/train_32x32.mat")
test = sio.loadmat("/share/data/vision-greg/svhn/test_32x32.mat")

#extra_32x32.mat
train_X = train['X'].reshape([32*32*3,73257])/255.
train_X = np.swapaxes(train_X, 0,1)
train_X = np.reshape(train_X, [73257, 32, 32, 3])
test_X = test['X'].reshape([32*32*3,26032])/255.
test_X = np.swapaxes(test_X, 0,1)

train_Y = train['y']
test_Y = test['y']

train_Y[train_Y==10] = 0
test_Y[test_Y==10] = 0

#make labels one hot
n_val = np.max(train['y']) + 1
train_temp = np.eye(n_val)[train_Y]
train_Y = train_temp.squeeze()
test_temp = np.eye(n_val)[test_Y]
test_Y = test_temp.squeeze()
