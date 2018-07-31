import tensorflow as tf
import numpy as np


class AlexNet:
    def __init__(self, number_of_classes):
        self.input_images = tf.placeholder(shape=[None, 227, 227, 3], dtype=tf.float64, name="input_images", )

        conv_1_kernel = tf.Variable(np.random.sample((11, 11,96)), dtype=tf.float64, name="conv_1_kernel")
        conv_1 = tf.nn.conv2d(self.input_images, conv_1_kernel, strides=(4, 4, 4, 4), dtype=tf.float64, name="conv_1")
        conv_1_activation = tf.nn.relu(conv_1, name="conv_1_activation")
        conv_1_max_pool = tf.nn.max_pool(conv_1_activation, ksize=(3, 3, 3, 3), name="conv_1_max_pool")

        conv_2_kernel = tf.Variable(np.random.sample((5, 5,256)), name="conv_2_kernel")
        conv_2 = tf.nn.conv2d(conv_1_max_pool, conv_2_kernel, name="conv_2")
        conv_2_activation = tf.nn.relu(conv_2, name="conv_2_activation")
        conv_2_max_pool = tf.nn.max_pool(conv_2_activation,)

    def fit(self, images, labels):
        pass

    def predict(self, x):
        with tf.Session() as session:
            return session.run(self.model, feed_dict={self.x: x})

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)
