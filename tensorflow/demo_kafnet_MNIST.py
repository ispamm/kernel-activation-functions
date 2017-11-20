# -*- coding: utf-8 -*-

# General imports
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# Custom imports
from kafnets import KAFNet

# Load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Initialize the KafNet
kafnet = KAFNet(obs_dim=784, n_class=10, h_size=64, n_layers=2, gamma=1.)

# General parameters
BATCH_SIZE = 64
MAX_STEPS = 10000
losses = []

saver = tf.train.Saver(var_list=kafnet._get_params())
with kafnet.sess.as_default():
    for idx in range(MAX_STEPS):
        # Train step
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        loss = kafnet.train(x=batch_xs, y=batch_ys)
        losses.append(loss)
        if idx % 100 == 0:
            print('Iteration: ', idx, ', loss: ', loss)

    scores = []

    batch_xs, batch_ys = mnist.validation.next_batch(BATCH_SIZE)
    score = kafnet.score(x=batch_xs, y=batch_ys)
    print('test: ', score)
    saver.save(sess=kafnet.sess, save_path='logs/model.ckpt')

kafnet.sess.close()
plt.plot(losses)
plt.savefig('img/train_loss.png')
plt.close()
