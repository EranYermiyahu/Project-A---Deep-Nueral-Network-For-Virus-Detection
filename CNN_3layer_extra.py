import numpy as np
import tensorflow as tf
from tensorflow import keras


class CNNExtended(keras.Model):
    def __init__(self, input_shape, num_classes, name='CNN_extra'):
        # Initialize the father -  requires to implement abstracts
        super(CNNExtended, self).__init__(name=name)
        # Layers
        self.num_classes = num_classes

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (9, 4), activation='relu', input_shape=input_shape, kernel_initializer='he_normal'),
            # 32*142*1
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(64, (15, 1), activation='relu', kernel_initializer='he_normal'),
            # 64*128*1
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
            # 64*64*1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (15, 1), activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dropout(0.1),
            # 128*50*1
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
            # 128*25*1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax',
                         activity_regularizer=tf.keras.regularizers.L2(0.1))
        ])


        # # self.conv1_layer = tf.keras.layers.Conv2D(20, (4, 4), activation="relu", input_shape=input_shape)
        # self.conv1_layer = tf.keras.layers.Conv2D(32, (9, 4), activation="relu", input_shape=input_shape)
        # # 32*142*1
        # self.conv2_layer = tf.keras.layers.Conv2D(64, (15, 1), activation="relu")
        # # 64*128*1
        # self.max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # self.Batchnorm = tf.keras.layers.BatchNormalization()
        # # 64*64*1
        # self.conv3_layer = tf.keras.layers.Conv2D(128, (15, 1), activation="relu")
        # # 128*50*1
        # self.flatter_layer = tf.keras.layers.Flatten()
        # self.CNN_output_layer = keras.layers.Dense(self.num_classes, activation='softmax',
        #                                            activity_regularizer=tf.keras.regularizers.L2(0.1))

        # self.flatter_layer = tf.keras.layers.Flatten(input_shape=input_shape)
        # self.middle_layer_1 = keras.layers.Dense(200, activation='relu')
        # self.middle_layer_2 = keras.layers.Dense(200, activation='relu')
        # self.linear_logistic_reg_layer = keras.layers.Dense(self.num_classes, activation='softmax',
        #                                                     activity_regularizer=tf.keras.regularizers.L2(0.2))
        self.model_name = name

    def call(self, inputs):
        return self.model(inputs)
