# -*- coding: utf-8 -*-

"""
Simple demo using kernel activation functions with convolutional networks on the MNIST dataset.
"""

# Keras imports
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
import keras.backend as K

# Custom imports
from kafnets import KAF

# Load Breast Cancer dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Preprocessing is taken from here:
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

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

# Training
kafnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
kafnet.summary()
kafnet.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Evaluation
print('Final accuracy is: ' + str(kafnet.evaluate(X_test, y_test, batch_size=64)[1]))