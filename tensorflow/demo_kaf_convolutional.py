# -*- coding: utf-8 -*-

"""
Simple demo using kernel activation functions with convolutional networks on the MNIST dataset.
"""

# Import TensorFlow
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

# Keras imports
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Custom imports
from kafnets import KAF
import tqdm

# Load Breast Cancer dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Preprocessing is taken from here:
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# Initialize a KAF neural network
kafnet = Sequential()
kafnet.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
kafnet.add(KAF(32, conv=True))
kafnet.add(Conv2D(32, (3, 3)))
kafnet.add(KAF(32, conv=True))
kafnet.add(MaxPooling2D(pool_size=(2, 2)))
kafnet.add(Flatten())
kafnet.add(Dense(100))
kafnet.add(KAF(100))
kafnet.add(Dense(10, activation='softmax'))

# Use tf.data DataLoader
train_data = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.int64)))
test_data = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), y_test.astype(np.int64)))

# Optimizer
opt = tf.train.AdamOptimizer()

# Training
for e in tqdm.trange(5, desc='Training'):
    
    for xb, yb in train_data.shuffle(1000).batch(32):
        
        with tfe.GradientTape() as tape:
            loss = tf.losses.sparse_softmax_cross_entropy(yb, kafnet(xb))
        g = tape.gradient(loss, kafnet.variables)
        opt.apply_gradients(zip(g, kafnet.variables))

    # Evaluation
    acc = tfe.metrics.Accuracy()
    for xb, yb in test_data.batch(32):
        acc(yb, tf.argmax(kafnet(xb), axis=1))
    tqdm.tqdm.write('Test accuracy after epoch {} is: '.format(e+1) + str(acc.result()))