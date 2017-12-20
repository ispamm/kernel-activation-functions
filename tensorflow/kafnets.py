# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class KAFNet(object):
    """
    KAF layer using kernel activation functions.
    """

    def __init__(self, obs_dim, n_class=1, h_size=4, n_layers=2, layer_type="linear", kernel='rbf', gamma=1.,
                 dict_size=20, boundary=3.0, beta=1e-5, lr=1e-3):
        self.D = tf.linspace(start=-boundary, stop=boundary, num=dict_size)
        self.scope = kernel
        self._init_ph(obs_dim=obs_dim, n_class=n_class)
        self._init_graph(h_size=h_size, n_layers=n_layers, layer_type=layer_type, kernel=kernel, n_class=n_class,
                         gamma=gamma)
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

    def _init_graph(self, h_size, n_layers, layer_type="linear", kernel="rbf", n_class=1, gamma=1.):
        h = tf.reshape(self.x, shape=(-1, 28, 28, 1)) if layer_type == "conv" else self.x
        self.alphas = []
        with tf.variable_scope(self.scope):
            for idx in range(n_layers):
                if layer_type == "conv":
                    out = tf.layers.conv2d(inputs=h, filters=h_size, kernel_size=5, strides=(2, 2),
                                           activation=None, name="conv_{}".format(idx),
                                           kernel_initializer=tf.random_normal_initializer(stddev=.1))
                else:
                    out = tf.layers.dense(inputs=h, units=h_size, activation=None, name='h_{}'.format(idx),
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.1))
                h, alpha = self.kaf(linear=out, name="kaf{}".format(idx), D=self.D, gamma=gamma, kernel=kernel)
                self.alphas.append(alpha)
            h = tf.contrib.layers.flatten(h) if layer_type == "conv" else h

            self.output = tf.layers.dense(inputs=h, units=n_class, activation=None, name='output',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            self.y_hat = tf.argmax(tf.nn.softmax(self.output), axis=1)

    @staticmethod
    def gauss_kernel(x, D, gamma=1.):

        x = tf.expand_dims(x, axis=-1)
        if x.get_shape().ndims < 4:
            # x = tf.reshape(x, (-1, tf.shape(x)[1], 1))
            # TODO there should be a better expression for this
            D = tf.reshape(D, (1, 1, -1))
        else:
            D = tf.reshape(D, (1, 1, 1, 1, -1))

        gauss_kernel = tf.exp(- gamma * tf.square(x - D))
        return gauss_kernel

    @staticmethod
    def gauss_kernel2D(x, Dx, Dy, gamma=1.):

        h_size = (x.get_shape()[-1].value) // 2

        x = tf.expand_dims(x, axis=-1)
        if x.get_shape().ndims < 4:
            Dx = tf.reshape(Dx, (1, 1, -1))
            Dy = tf.reshape(Dy, (1, 1, -1))
            x1, x2 = x[:, :h_size], x[:, h_size:]
        else:
            Dy = tf.reshape(Dy, (1, 1, 1, 1, -1))
            Dx = tf.reshape(Dx, (1, 1, 1, 1, -1))
            x1, x2 = x[:, :, :, :h_size], x[:, :, :, h_size:]
        gauss_kernel = tf.exp(-gamma * tf.square(x1 - Dx)) + tf.exp(- gamma * tf.square(x2 - Dy))
        return gauss_kernel

    @staticmethod
    def kaf(linear, name, kernel='rbf', D=None, gamma=1., ):

        if D is None:
            D = tf.linspace(start=-2., stop=2., num=20)

        with tf.variable_scope('kaf'):
            if kernel == 'rbf':
                K = KAFNet.gauss_kernel(linear, D, gamma=gamma)
                alpha = tf.get_variable(name, shape=(1, linear.get_shape()[-1], D.get_shape()[0]),
                                        initializer=tf.random_normal_initializer(stddev=0.1))
            elif kernel == 'rbf2d':
                Dx, Dy = tf.meshgrid(D, D)
                K = KAFNet.gauss_kernel2D(linear, Dx, Dy, gamma=gamma)

                alpha = tf.get_variable(name,
                                        shape=(1, linear.get_shape()[-1] // 2, D.get_shape()[0] * D.get_shape()[0]),
                                        initializer=tf.random_normal_initializer(stddev=0.1))
            else:
                raise NotImplementedError()
            act = tf.reduce_sum(tf.multiply(K, alpha), axis=-1)
        return (act, alpha)

    @staticmethod
    def compute_accuracy_op(y, y_hat):
        correct_predictions = tf.equal(tf.argmax(y, axis=1), y_hat)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return accuracy
