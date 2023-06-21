import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


class Statistics():
    def __init__(self, checkpoint_path, model):
        self.model = (tf.train.Checkpoint(model=model)).restore(checkpoint_path).expect_partial()

    def accuracy_graph(self, inputs):
        # middle = self.middle_layer(self.flatter_layer(inputs))
        middle_1 = self.flatter_layer(inputs)
        middle_2 = self.middle_layer_1(middle_1)

        return self.linear_logistic_reg_layer(middle_2)
