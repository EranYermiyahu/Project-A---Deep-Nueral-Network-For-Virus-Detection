
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
    def __init__(self, virus_list=["Coronaviridae", "InfluenzaA", "Metapneumovirus", "Rhinovirus", "SarsCov2"], fragment_size=150):
        self.Viruses_list = virus_list
        self.viruses_num = len(virus_list)
        self.fragment_size = fragment_size
        # self.train_tfr_paths = None
        self.nuc_translation_dictionary = {
            "A": np.array([1, 0, 0, 0], dtype=np.int8),
            "G": np.array([0, 1, 0, 0], dtype=np.int8),
            "C": np.array([0, 0, 1, 0], dtype=np.int8),
            "T": np.array([0, 0, 0, 1], dtype=np.int8),
            "N": np.array([0, 0, 0, 0], dtype=np.int8)
        }
        self.virus_label_dictionary = None
        # This element contains array of arrays - by each virus- the whole segments
        self.all_segments_by_virus = None
        # self.all_fragments_by_virus = None
        self.all_tokened_frags_by_virus = None

    def set_all_segments_by_virus(self, seg_by_vir_list):
        self.all_segments_by_virus = np.asarray(seg_by_vir_list)

    def set_all_tokened_frags_by_virus(self, tokened_frags_by_vir_list):
        self.all_tokened_frags_by_virus = np.asarray(tokened_frags_by_vir_list)

    def create_labels_from_virus_list(self):
        if self.Viruses_list is None:
            raise Exception("No Viruses list has given. Please insert list on init")
        pos = 0
        label_dict = {}
        for virus in self.Viruses_list:
            label_dict[virus] = np.array([1 if i == pos else 0 for i in range(self.viruses_num)], dtype=np.int8)
            pos += 1
        self.virus_label_dictionary = label_dict
        print(self.virus_label_dictionary)

    def segments_from_fna_file(self, fna_file_name):
        cur_file = open(fna_file_name, 'r')
        raw_text = cur_file.read()
        # Parse raw data into np of segments
        raw_segment_list = raw_text.split('>')
        raw_segments = (np.asarray(raw_segment_list))[1:]
        raw_segments = np.char.partition(raw_segments, '\n')[:, 2]
        segments_per_file = np.char.replace(raw_segments, '\n', '')
        return segments_per_file

    def start_idx_to_fragment(self, num, seg):
        return seg[num:num + self.fragment_size]

    def one_segment_to_fragments(self, seg):
        # For each segment create list of the valid indexes
        interval = len(seg) - self.fragment_size
        valid_index_per_seg_list = list(range(interval))
        # For each segment change the list of start indexes to numpy of fragments
        all_fragments_per_seg = np.asarray(list(map(self.start_idx_to_fragment, valid_index_per_seg_list, repeat(seg))))
        return all_fragments_per_seg

    # Return matrix num_of_frag X len(segments) with randomly fragments from the segments
    def generate_fragmnets_from_segments(self, segments):
        all_fragments_per_virus = np.concatenate(list(map(self.one_segment_to_fragments, segments)), axis=0)
        all_tokened_frags = np.asarray(list(map(self.fragment_to_token, all_fragments_per_virus)))
        # In order to save the fragments inside the class - Need to implement the setter inside here locally per virus
        return all_tokened_frags

    def all_viruses_fragments(self):
        all_tokened_frags_by_virus_list = list(map(self.generate_fragmnets_from_segments, self.all_segments_by_virus))
        self.set_all_tokened_frags_by_virus(all_tokened_frags_by_virus_list)

    def char_to_vector(self, char):
        return self.nuc_translation_dictionary[char]

    def fragment_to_token(self, fragment):
        # Create np array of chars, each char is nucleotide
        broken_frag = np.asarray([char for char in fragment])
        # Create fragsize X 4 matrix from each nucleotide, using map function for converting
        frag_Tensor = np.asarray(list(map(self.char_to_vector, broken_frag.T)))
        return frag_Tensor

    def segment_creation_from_fna(self):
        segments_virus_list = []
        # Iterate each virus and create np array of segments
        for CurrentVirus in self.Viruses_list:
            # List of all fna name files per virus
            fna_files_list = glob.glob("../Viruses Raw Data/{}/ncbi_dataset/data/GCA**/*.fna".format(CurrentVirus))
            # Iterate each file according to virus and extract segments into np array
            segment_per_virus = np.concatenate(list(map(self.segments_from_fna_file, fna_files_list)))
            # Create list of all np arrays by virus
            segments_virus_list.append(segment_per_virus)
        # This element contains array of arrays - by each virus- the whole segments
        self.set_all_segments_by_virus(segments_virus_list)

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

    def create_single_path(self, virus_name):
        return '../TFRecords/{}.tfrecord'.format(virus_name)

    def write_fragments_to_tfr(self, fragments, label, tfr_file_path, shuffle=False):
        Serialized_Tensor_list = list(map(self.serialized_tensor, fragments, repeat(label)))
        if shuffle is True:
            random.shuffle(Serialized_Tensor_list)
        with tf.io.TFRecordWriter(tfr_file_path) as f:
            for example in Serialized_Tensor_list:
                f.write(example.SerializeToString())

    def viruses_to_tfr_files(self):
        for virus_idx in range(self.viruses_num):
            tokened_frags = self.all_tokened_frags_by_virus[virus_idx]
            v_label = self.virus_label_dictionary[self.Viruses_list[virus_idx]]
            v_path = self.create_single_path(self.Viruses_list[virus_idx])
            self.write_fragments_to_tfr(tokened_frags, v_label, v_path)
