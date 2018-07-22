import torch

'''
 import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            
            return F.relu(self.conv2(x))

'''


class Alexnet(torch.nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(3, 96, 11, stride=4)

        self.pool_1 = torch.nn.MaxPool2d(3, stride=2)

        conv_2 = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), activation="relu")(norm_1)
        pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_2)
        norm_2 = tf.keras.layers.BatchNormalization()(pool_2)

        conv_3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu")(norm_2)
        conv_4 = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), activation="relu")(conv_3)
        conv_5 = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), activation="relu")(conv_4)

        pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_5)
        norm_3 = tf.keras.layers.BatchNormalization()(pool_3)

        flat1 = tf.keras.layers.Flatten()(norm_3)
        dense_1 = tf.keras.layers.Dense(4096, activation="tanh")(flat1)
        dropout_1 = tf.keras.layers.Dropout(0.5)(dense_1)

        dense_2 = tf.keras.layers.Dense(4096, activation="tanh")(dropout_1)
        dense_3 = tf.keras.layers.Dropout(0.5)(dense_2)

        output = tf.keras.layers.Dense(17, activation="softmax")(dense_3)

        self.model = tf.keras.Model(inputs=input, outputs=output)
        self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
