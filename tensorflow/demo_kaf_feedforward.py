# -*- coding: utf-8 -*-

"""
Simple demo using kernel activation functions on a basic regression dataset.
"""

# Import TensorFlow
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

# Keras imports
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Custom imports
from kafnets import KAF
import tqdm

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

# Use tf.data DataLoader
train_data = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train.reshape(-1, 1)))
test_data = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), y_test.astype(np.float32).reshape(-1, 1)))

# Optimizer
opt = tf.train.AdamOptimizer()

# Training
for e in tqdm.trange(300, desc='Training'):
    
    for xb, yb in train_data.shuffle(1000).batch(32):
        
        with tfe.GradientTape() as tape:
            loss = tf.losses.mean_squared_error(yb, kafnet(xb))
        g = tape.gradient(loss, kafnet.variables)
        opt.apply_gradients(zip(g, kafnet.variables))

# Evaluation
err = tfe.metrics.Mean()
for xb, yb in test_data.batch(32):
    err((yb - kafnet(xb))**2)
print('Final error is: ' + str(err.result()))
