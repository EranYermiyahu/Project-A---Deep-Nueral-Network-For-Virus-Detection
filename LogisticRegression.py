import numpy as np
import tensorflow as tf
from tensorflow import keras



class LogisticRegression(keras.Model):
    def __init__(self, input_shape, num_classes):
        # Initialize the father -  requires to implement abstracts
        super(LogisticRegression, self).__init__(name='LogisticRegression')
        # Layers
        self.num_classes = num_classes
        self.flatter_layer = tf.keras.layers.Flatten(input_shape=input_shape)
        self.linear_logistic_reg_layer = keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs):
        middle = self.flatter_layer(inputs)
        return self.linear_logistic_reg_layer(middle)



