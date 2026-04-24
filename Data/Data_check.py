import numpy as np
import pickle

X_train = np.load('processed/X_train.npy')
X_val   = np.load('processed/X_val.npy')
X_test  = np.load('processed/X_test.npy')

print('X_train:', X_train.shape)
print('X_val:  ', X_val.shape)
print('X_test: ', X_test.shape)