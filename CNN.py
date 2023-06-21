import numpy as np
import tensorflow as tf
from tensorflow import keras


class CNN(keras.Model):
    def __init__(self, input_shape, num_classes, name='CNN_Tuned'):
        # Initialize the father -  requires to implement abstracts
        super(CNN, self).__init__(name=name)
        # Layers
        self.num_classes = num_classes

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (9, 4), activation='relu', input_shape=input_shape, kernel_initializer='he_normal'),
            # 32*142*1
            tf.keras.layers.Conv2D(64, (15, 1), activation='relu', kernel_initializer='he_normal'),
            # 64*128*1
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
            # 64*64*1
            tf.keras.layers.Conv2D(128, (15, 1), activation='relu', kernel_initializer='he_normal'),
            # 128*50*1
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
            # 128*25*1
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model_name = name

    def call(self, inputs):
        return self.model(inputs)
