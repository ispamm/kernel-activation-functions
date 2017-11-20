
import os
import numpy as np
import tensorflow as tf

"""
Some utility functions.
"""

def fc(x, h_size, name, act=None, std=0.1):
    # Fully connected layer
    with tf.variable_scope(name):
        input_size = x.get_shape()[1]
        w = tf.get_variable('w', (input_size, h_size), initializer=tf.random_normal_initializer(stddev=std))
        b = tf.get_variable('b', (h_size), initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        if act is not None:
            z = act(z)
        return z

def set_global_seed(seed=1):
    # Set random seed globally
    try:
        np.random.seed(seed)
        tf.set_random_seed(seed)
    except:
        pass

def _save(saver, sess, log_dir):
    # Save current session
    try:
        saver.save(sess=sess, save_path=os.path.join(log_dir, 'model.ckpt'))
    except Exception as e:
        tf.logging.error(e)
        raise e

def _load(saver, sess, log_dir):
    # Load current session
    try:
        ckpt = tf.train.latest_checkpoint(log_dir)
        saver.restore(sess=sess, save_path=ckpt)
    except Exception as e:
        tf.logging.error(e)
        raise e
