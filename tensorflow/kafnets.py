# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class KAFNet(object):
    """
    KAF layer using kernel activation functions.
    """

    def __init__(self, obs_dim, n_class=1, h_size=4, n_layers=2, gamma=1., kernel='rbf', dict_size=20, boundary=3.0,
                 beta=1e-5, lr=1e-3):
        self.D = tf.linspace(start=-boundary, stop=boundary, num=dict_size)
        self.scope = kernel
        self._init_ph(obs_dim=obs_dim, n_class=n_class)
        self._init_graph(n_class=n_class, h_size=h_size, n_layers=n_layers, gamma=gamma, dict_size=dict_size)
        self._train_op(lr=lr, beta=beta)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, x, y):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.x: x, self.y: y})
        return loss

    def predict(self, x):
        if np.ndim(x) < 1:
            x = [x]
        y_hat = self.sess.run(self.y_hat, feed_dict={self.x: x})
        return y_hat

    def score(self, x, y):
        accuracy = self.sess.run(self.compute_accuracy_op(y=self.y, y_hat=self.y_hat),
                                 feed_dict={self.x: x, self.y: y})
        return accuracy

    def _init_ph(self, obs_dim, n_class):
        self.x = tf.placeholder(tf.float32, shape=(None, obs_dim), name='X')
        self.y = tf.placeholder(tf.float32, shape=(None, n_class), name='Y')

    def _train_op(self, lr=1e-3, beta=1e-5):
        l2_loss = beta * tf.add_n([tf.nn.l2_loss(t=alpha) for alpha in self.alphas])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y) + l2_loss)
        # self.loss = .5 * tf.reduce_mean(tf.square(self.y - self.y_hat), name='mse')
        optim = tf.train.AdamOptimizer(learning_rate=lr)
        grads = tf.gradients(self.loss, self._get_params())
        self.train_op = optim.apply_gradients(zip(grads, self._get_params()))

    def _get_params(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def _init_graph(self, h_size, n_layers, n_class=1, gamma=1., dict_size=20):
        h = self.x
        self.alphas = []
        with tf.variable_scope(self.scope):
            for idx in range(n_layers):
                alpha = tf.get_variable('alpha_{}'.format(idx), shape=(h_size, dict_size),
                                        initializer=tf.random_normal_initializer(stddev=0.1))
                self.alphas.append(alpha)
                linear = tf.layers.dense(inputs=h, units=h_size, activation=None, name='h_{}'.format(idx),
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.1))
                h = self.kaf(linear=linear, D=self.D, alpha=alpha, gamma=gamma)

            self.output = tf.layers.dense(inputs=h, units=n_class, activation=None, name='output',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            self.y_hat = tf.argmax(tf.nn.softmax(self.output), axis=1)

    @staticmethod
    def gauss_kernel(x, D, gamma=1.):
        gauss_kernel = tf.exp(
            - gamma * tf.square(tf.reshape(x, (-1, tf.shape(x)[1], 1)) - tf.reshape(D, (1, 1, -1))))
        return gauss_kernel

    @staticmethod
    def kaf(linear, kernel='rbf', D=None, alpha=None, gamma=1.):

        if D is None:
            D = tf.linspace(start=-2., stop=2., num=20)

        with tf.variable_scope('kaf'):
            if kernel == 'rbf':
                K = KAFNet.gauss_kernel(linear, D, gamma=gamma)
            else:
                raise NotImplementedError()

            alpha = tf.reshape(alpha, (1, linear.get_shape()[1].value, tf.shape(D)[0]))
        return tf.reduce_sum(tf.multiply(K, alpha), axis=2)

    @staticmethod
    def compute_accuracy_op(y, y_hat):
        correct_predictions = tf.equal(tf.argmax(y, axis=1), y_hat)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return accuracy
