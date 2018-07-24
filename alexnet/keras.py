import tensorflow as tf


class AlexNet():
    def __init__(self, number_of_classes):
        input_net = tf.keras.Input(shape=(227, 227, 3))

        conv_1 = tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation="relu")(input)
        pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_1)

        conv_2 = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), activation="relu")(pool_1)
        pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_2)

        conv_3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu")(pool_2)
        conv_4 = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), activation="relu")(conv_3)
        conv_5 = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), activation="relu")(conv_4)

        pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_5)

        flat1 = tf.keras.layers.Flatten()(pool_3)
        dense_1 = tf.keras.layers.Dense(4096, activation="tanh")(flat1)
        dropout_1 = tf.keras.layers.Dropout(0.5)(dense_1)

        dense_2 = tf.keras.layers.Dense(4096, activation="tanh")(dropout_1)
        dense_3 = tf.keras.layers.Dropout(0.5)(dense_2)

        output = tf.keras.layers.Dense(number_of_classes, activation="softmax")(dense_3)

        self.model = tf.keras.Model(inputs=input_net, outputs=output)
        self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    def fit(self, images, labels):
        self.model.fit(x=images, y=labels)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)
