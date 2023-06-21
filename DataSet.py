
import numpy as np
import tensorflow as tf
import os
import os.path
import glob
from itertools import repeat
import random


class DataSet:
    def __init__(self, virus_list, data_files_path='../TFRecords', fragment_size=150,
                 nuc_to_token_base=4, train_percentage=0.9):
        self.Viruses_list = virus_list
        self.viruses_num = len(virus_list)
        self.fragment_size = fragment_size
        self.nuc_base_count = nuc_to_token_base
        self.tfr_paths_list_train = [data_files_path + '/Train' + virus + '.tfrecord' for virus in self.Viruses_list]
        self.tfr_paths_list_test = [data_files_path + '/Test' + virus + '.tfrecord' for virus in self.Viruses_list]
        self.train_per = train_percentage
        self.train_dataset_size = None
        self.test_dataset_size = None

    def get_train_tfr_paths(self):
        return self.tfr_paths_list_train

    def get_test_tfr_paths(self):
        return self.tfr_paths_list_test

    def config_data_features(self, raw_dataset, epochs=1, batch_size=128, shuffle_buffer_size=4096):
        config_data_set = raw_dataset.repeat(epochs). \
                    shuffle(shuffle_buffer_size). \
                    batch(batch_size). \
                    prefetch(tf.data.AUTOTUNE)
        return config_data_set

    def create_train_dataset(self, epochs=1, train_batch_size=4096, shuffle_buffer_size=4096):
        filepath_dataset = tf.data.Dataset.list_files(self.get_train_tfr_paths())
        train_dataset_raw = filepath_dataset.interleave(map_func=lambda filepath: tf.data.TFRecordDataset(filepath),
                                                        num_parallel_calls=tf.data.AUTOTUNE,
                                                        cycle_length=self.viruses_num). \
            map(map_func=lambda example_proto: self.deserialize_genome_tensor(example_proto),
                num_parallel_calls=tf.data.AUTOTUNE)

        self.train_dataset_size = int(train_dataset_raw.reduce(np.int64(0), lambda x, _: x + 1))
        train_dataset = self.config_data_features(train_dataset_raw, epochs=epochs, batch_size=train_batch_size,
                                                  shuffle_buffer_size=shuffle_buffer_size)
        return train_dataset

    def create_test_dataset(self, test_batch_size=128, shuffle_buffer_size=4096):
        filepath_dataset = tf.data.Dataset.list_files(self.get_test_tfr_paths())
        test_data_set_raw = filepath_dataset.interleave(map_func=lambda filepath: tf.data.TFRecordDataset(filepath),
                                                        num_parallel_calls=tf.data.AUTOTUNE,
                                                        cycle_length=self.viruses_num). \
            map(map_func=lambda example_proto: self.deserialize_genome_tensor(example_proto),
                num_parallel_calls=tf.data.AUTOTUNE)

        self.test_dataset_size = int(test_data_set_raw.reduce(np.int64(0), lambda x, _: x + 1))
        test_dataset = self.config_data_features(test_data_set_raw, batch_size=test_batch_size,
                                                 shuffle_buffer_size=shuffle_buffer_size)
        return test_dataset

    def print_labels_histogram(self, dataset, virus_library):
        counter = 0
        for tokens, labels in dataset:
            how_much_labels_in_batch_dict = {"Coronaviridae": 0, "InfluenzaA": 0, "Metapneumovirus": 0, "Rhinovirus": 0,
                                             "SarsCov2": 0}
            # ret_val = model.call(tokens)
            for label in labels:
                for key, val in virus_library.items():
                    print(f"{val}\n{label}")
                    val_virus_idx = int(tf.argmax(val))
                    label_virus_idx = int(tf.argmax(label))
                    if val_virus_idx == label_virus_idx:
                        how_much_labels_in_batch_dict[key] += 1
                        break
            counter = counter+1
            if counter >= 1000:
                print("hi")
            print(how_much_labels_in_batch_dict)

    def write_fragments_to_tfr(self, fragments, label, tfr_file_path, shuffle=False):
        Serialized_Tensor_list = list(map(self.serialized_tensor, fragments, repeat(label)))
        if shuffle is True:
            random.shuffle(Serialized_Tensor_list)
        with tf.io.TFRecordWriter(tfr_file_path) as f:
            for example in Serialized_Tensor_list:
                f.write(example.SerializeToString())
    # For a given path - convert the token,label to example and write it to path

    def create_tfrecords(self, list_token_frags, list_labels):
        for virus_idx in range(self.viruses_num):
            train_size = int(self.train_per * len(list_token_frags[virus_idx]))
            self.write_fragments_to_tfr(list_token_frags[virus_idx][:train_size],
                                        list_labels[virus_idx], self.tfr_paths_list_train[virus_idx])
            self.write_fragments_to_tfr(list_token_frags[virus_idx][train_size + 1:],
                                        list_labels[virus_idx], self.tfr_paths_list_test[virus_idx])
    # Send the relevant tokens and label to the TFR creator

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


