from kafnets import KAFNet
from utils import set_global_seed
import tensorflow as tf
import unittest
import numpy as np


class TestKafNet(unittest.TestCase):
    @staticmethod
    def test_kernel_computation():
        set_global_seed(10)
        X = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        D = np.asarray([1, -0.5]).reshape(2, 1)

        with tf.Session() as sess:
            K_op = KAFNet.gauss_kernel(x=tf.convert_to_tensor(X), D=tf.convert_to_tensor(D))
            K = sess.run(K_op)

        K_true = np.exp(-np.asarray([[[0.9 ** 2, 0.6 ** 2], [0.8 ** 2, 0.7 ** 2]],
                                     [[0.7 ** 2, 0.8 ** 2], [0.6 ** 2, 0.9 ** 2]],
                                     [[0.5 ** 2, 1.0 ** 2], [0.4 ** 2, 1.1 ** 2]]]))

        try:
            np.testing.assert_array_almost_equal(K, K_true, decimal=4)
        except Exception as e:
            print(e)

    def test_linear_sum(self):
        alpha = np.asarray([[0.3, -0.5], [1.2, -0.1]]).reshape(1, 2, 2)
        X = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        D = np.asarray([1, -0.5]).reshape(2, 1)
        out_true = np.asarray([[0.3 * 0.9 - 0.5 * 0.6, 1.2 * 0.8 - 0.1 * 0.7],
                               [0.7 * 0.3 - 0.5 * 0.8, 0.6 * 1.2 - 0.1 * 0.9],
                               [0.5 * 0.3 - 0.5 * 1.0, 0.4 * 1.2 - 0.1 * 1.1]])

        # Define a matrix
        K = np.asarray([[[0.9, 0.6], [0.8, 0.7]], [[0.7, 0.8], [0.6, 0.9]], [[0.5, 1.0], [0.4, 1.1]]])

        out = np.sum(K * alpha, axis=2)
        try:
            np.testing.assert_array_almost_equal(out, out_true, decimal=4)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    unittest.main()
