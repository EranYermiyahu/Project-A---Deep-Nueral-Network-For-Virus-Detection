import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import tensorflow_datasets as tfds


class LogisticRegression(keras.Model):
    def __init__(self, num_of_classes):
        # Initialize the father -  requires to implement abstracts
        super(LogisticRegression, self).__init__(name='LogisticRegression')
        # Layers
        self.linear_logistic_reg_layer = keras.layers.Dense(num_of_classes, activation='softmax')

    def call(self, inputs):
        return self.linear_logistic_reg_layer(inputs)





