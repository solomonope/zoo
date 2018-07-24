import tensorflow as tf


class AlexNet():
    def __init__(self, number_of_classes):
        self.x = tf.placeholder(name="x", dtype=tf.int64)
        self.model = tf.add(tf.constant(5, name="roc", dtype=tf.int64), self.x)

    def fit(self, images, labels):
        pass

    def predict(self, x):
        with tf.Session() as session:
            return session.run(self.model, feed_dict={self.x: x})

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)
