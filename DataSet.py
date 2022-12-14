
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import os
import os.path
import glob
from itertools import repeat
import random


class DataSet:
    def __init__(self, virus_list, data_files_path='../TFRecords'):
        self.Viruses_list = virus_list
        self.viruses_num = len(virus_list)
        self.tfr_paths_list = [data_files_path + '/' + virus + '.tfrecord' for virus in self.Viruses_list]



    def serialized_tensor(self, tensor: tf.Tensor, label: tf.Tensor) -> tf.train.Example:
        # Serialize the tensor
        serialized_tensor = tf.io.serialize_tensor(tensor)
        serialized_label = tf.io.serialize_tensor(label)

        # Store the data in a tf.train.Feature (which is a protobuf object)
        feature_of_bytes_tensor = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialized_tensor.numpy()])
        )
        feature_of_bytes_label = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialized_label.numpy()])
        )

        # Put the tf.train.Feature message into a tf.train.Example (which is a protobuf object that will be written into the file)
        features_for_example = {
            'tensor': feature_of_bytes_tensor,
            'label': feature_of_bytes_label
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=features_for_example)
        )
        return example_proto

    # def create_single_path(self, virus_name):
    #     return self.data_files_path + '/{}.tfrecord'.format(virus_name)

    def write_fragments_to_tfr(self, fragments, label, tfr_file_path, shuffle=False):
        Serialized_Tensor_list = list(map(self.serialized_tensor, fragments, repeat(label)))
        if shuffle is True:
            random.shuffle(Serialized_Tensor_list)
        with tf.io.TFRecordWriter(tfr_file_path) as f:
            for example in Serialized_Tensor_list:
                f.write(example.SerializeToString())

    def create_tfrecords(self, list_token_frags, list_labels):
        for virus_idx in range(self.viruses_num):
            self.write_fragments_to_tfr(list_token_frags[virus_idx], list_labels[virus_idx], self.tfr_paths_list[virus_idx])


