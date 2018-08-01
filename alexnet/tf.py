import tensorflow as tf
import numpy as np


class AlexNet:
    def __init__(self, number_of_classes):
        self.input_images = tf.placeholder(shape=[None, 227, 227, 3], dtype=tf.float64, name="input_images", )

        conv_1_kernel = tf.Variable(np.random.sample((11, 11, 3, 96)), dtype=tf.float64, name="conv_1_kernel")
        conv_1 = tf.nn.conv2d(self.input_images, conv_1_kernel, strides=(4, 4, 4, 4), dtype=tf.float64, name="conv_1")
        conv_1_activation = tf.nn.relu(conv_1, name="conv_1_activation")
        conv_1_max_pool = tf.nn.max_pool(conv_1_activation, ksize=(1, 3, 3, 1), name="conv_1_max_pool")

        conv_2_kernel = tf.Variable(np.random.sample((5, 5, 96, 256)), name="conv_2_kernel")
        conv_2 = tf.nn.conv2d(conv_1_max_pool, conv_2_kernel, name="conv_2")
        conv_2_activation = tf.nn.relu(conv_2, name="conv_2_activation")
        conv_2_max_pool = tf.nn.max_pool(conv_2_activation, ksize=(1, 3, 3, 1), name="conv_2_max_pool")

        conv_3_kernel = tf.Variable(np.random((3, 3, 256, 384)), name="conv_3_kernel")
        conv_3 = tf.nn.conv2d(conv_2_max_pool, conv_3_kernel, strides=(1, 1, 1, 1), name="conv_3")
        conv_3_activation = tf.nn.relu(conv_3, name="conv_3_activation")

        conv_4_kernel = tf.Variable(np.random((3, 3, 256, 384)), name="conv_4_kernel")
        conv_4 = tf.nn.conv2d(conv_3_activation, conv_4_kernel, strides=(1, 1, 1, 1), name="conv_4")
        conv_4_activation = tf.nn.relu(conv_4, name="conv_3_activation")

        conv_5_kernel = tf.Variable(np.random((3, 3, 256, 384)), name="conv_5_kernel")
        conv_5 = tf.nn.conv2d(conv_4_activation, conv_5_kernel, strides=(1, 1, 1, 1), name="conv_5")
        conv_5_activation = tf.nn.relu(conv_5, name="conv_3_activation")

        conv_5_max_pool = tf.nn.max_pool(conv_5_activation, ksize=(1, 3, 3, 1), name="conv_5_activation")

        tf.nn.fla

    def fit(self, images, labels):
        pass

    def predict(self, x):
        with tf.Session() as session:
            return session.run(self.model, feed_dict={self.x: x})

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)
