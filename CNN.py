import numpy as np
import tensorflow as tf
from tensorflow import keras


class CNN(keras.Model):
    def __init__(self, input_shape, num_classes):
        # Initialize the father -  requires to implement abstracts
        super(CNN, self).__init__(name='CNN')
        # Layers
        self.num_classes = num_classes
        self.conv1_layer = tf.keras.layers.Conv2D(50, (4, 4), activation="relu", input_shape=input_shape)
        self.conv2_layer = tf.keras.layers.Conv2D(10, (10, 1), activation="relu")
        self.flatter_layer = tf.keras.layers.Flatten()
        self.CNN_output_layer = keras.layers.Dense(self.num_classes, activation='softmax',
                                                   activity_regularizer=tf.keras.regularizers.L2(0.2))

        # self.flatter_layer = tf.keras.layers.Flatten(input_shape=input_shape)
        # self.middle_layer_1 = keras.layers.Dense(200, activation='relu')
        # self.middle_layer_2 = keras.layers.Dense(200, activation='relu')
        # self.linear_logistic_reg_layer = keras.layers.Dense(self.num_classes, activation='softmax',
        #                                                     activity_regularizer=tf.keras.regularizers.L2(0.2))
        self.model_name = "CNN"

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=-1)
        # middle = self.middle_layer(self.flatter_layer(inputs))
        middle_1 = self.conv1_layer(inputs)
        middle_2 = self.conv2_layer(middle_1)
        middle_3 = self.flatter_layer(middle_2)

        return self.CNN_output_layer(middle_3)
