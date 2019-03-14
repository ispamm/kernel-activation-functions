# -*- coding: utf-8 -*-

"""
Simple demo using kernel activation functions on a basic regression dataset.
"""

# Keras imports
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense

# Custom imports
from kafnets import KAF

# Load Breast Cancer dataset
(X_train, y_train), (X_test, y_test) = datasets.boston_housing.load_data()

# Initialize a KAF neural network
kafnet = Sequential([
    Dense(20, input_shape=(13,)),
    KAF(20),
    Dense(1),
])

#Uncomment to use KAF with Softplus kernel
#kafnet = Sequential([
#    Dense(20, input_shape=(13,)),
#    KAF(20, kernel='softplus', D=5),
#    Dense(1),
#])

# Training
kafnet.compile(optimizer='adam', loss='mse')
kafnet.summary()
kafnet.fit(X_train, y_train, epochs=250, batch_size=32, verbose=0)

# Evaluation
print('Final error is: ' + str(kafnet.evaluate(X_test, y_test, batch_size=64)))