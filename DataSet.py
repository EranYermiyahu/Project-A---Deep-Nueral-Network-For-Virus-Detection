
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
    def __init__(self, virus_list, data_files_path='../TFRecords', fragment_size=150, nuc_to_token_base=4,\
                 train_per=0.95):
        self.Viruses_list = virus_list
        self.viruses_num = len(virus_list)
        self.fragment_size = fragment_size
        self.nuc_base_count = nuc_to_token_base
        self.tfr_paths_list = [data_files_path + '/' + virus + '.tfrecord' for virus in self.Viruses_list]
        self.valid_per = 1 - train_per
        self.train_per = train_per
        # self.epochs = epochs
        # self.batch_size = batch_size

    def create_train_dataset(self, epochs=1, batch_size=128, shuffle_buffer_size=4096):
        filepath_dataset = tf.data.Dataset.list_files(self.get_tfr_paths())
        train_set = filepath_dataset.interleave(map_func=lambda filepath: tf.data.TFRecordDataset(filepath),
                                                num_parallel_calls=tf.data.AUTOTUNE,
                                                cycle_length=self.viruses_num). \
            map(map_func=lambda example_proto: self.deserialize_genome_tensor(example_proto),
                num_parallel_calls=tf.data.AUTOTUNE). \
            repeat(epochs). \
            shuffle(shuffle_buffer_size). \
            batch(batch_size). \
            prefetch(tf.data.AUTOTUNE)
        return train_set

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

    def get_tfr_paths(self):
        return self.tfr_paths_list

    def write_fragments_to_tfr(self, fragments, label, tfr_file_path, shuffle=False):
        # print('~~~~~')
        Serialized_Tensor_list = list(map(self.serialized_tensor, fragments, repeat(label)))
        if shuffle is True:
            random.shuffle(Serialized_Tensor_list)
        with tf.io.TFRecordWriter(tfr_file_path) as f:
            for example in Serialized_Tensor_list:
                f.write(example.SerializeToString())

    def create_tfrecords(self, list_token_frags, list_labels):
        for virus_idx in range(self.viruses_num):
            self.write_fragments_to_tfr(list_token_frags[virus_idx], list_labels[virus_idx], self.tfr_paths_list[virus_idx])

    feature_description = {
        'tensor': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }

    def deserialize_genome_tensor(self, example_proto):
        feature_map = tf.io.parse_single_example(example_proto, self.feature_description)
        tensor_shape = [self.fragment_size, self.nuc_base_count]
        label_shape = [self.viruses_num]
        tensor = tf.ensure_shape(tf.io.parse_tensor(feature_map['tensor'], out_type=tf.int8), tensor_shape)
        tensor = tf.cast(tensor, tf.float32)
        label = tf.ensure_shape(tf.io.parse_tensor(feature_map['label'], out_type=tf.int8), label_shape)
        label = tf.cast(label, tf.float32)
        return tensor, label
