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
        self.middle_layer_1 = keras.layers.Dense(200, activation='relu')
        self.middle_layer_2 = keras.layers.Dense(200, activation='relu')
        self.linear_logistic_reg_layer = keras.layers.Dense(self.num_classes, activation='softmax',
                                                            activity_regularizer=tf.keras.regularizers.L2(0.2))
        self.model_name = "Logistic regression"

    def call(self, inputs):
        # middle = self.middle_layer(self.flatter_layer(inputs))
        middle_1 = self.flatter_layer(inputs)
        middle_2 = self.middle_layer_1(middle_1)

        return self.linear_logistic_reg_layer(middle_2)
