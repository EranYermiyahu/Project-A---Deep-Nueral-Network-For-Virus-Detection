
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import os
import os.path
import glob
from itertools import repeat
import random


class DNASeq:
    def __init__(self, virus_list=["Coronaviridae", "InfluenzaA", "Metapneumovirus", "Rhinovirus", "SarsCov2"],
                 fasta_files_path='../', fragment_size=150):
        self.Viruses_list = virus_list
        self.viruses_num = len(virus_list)
        self.fragment_size = fragment_size
        self.fasta_files_path = fasta_files_path
        # self.tensor_shape = (self.fragment_size, len(self.nuc_translation_dictionary['A'])
        # self.train_tfr_paths = None
        self.nuc_translation_dictionary = {
            "A": np.array([1, 0, 0, 0], dtype=np.int8),
            "G": np.array([0, 1, 0, 0], dtype=np.int8),
            "C": np.array([0, 0, 1, 0], dtype=np.int8),
            "T": np.array([0, 0, 0, 1], dtype=np.int8),
            "U": np.array([0, 0, 0, 1], dtype=np.int8),
            "W": np.array([1, 0, 0, 1], dtype=np.int8),
            "S": np.array([0, 1, 1, 0], dtype=np.int8),
            "M": np.array([1, 0, 1, 0], dtype=np.int8),
            "K": np.array([0, 1, 0, 1], dtype=np.int8),
            "R": np.array([1, 1, 0, 0], dtype=np.int8),
            "Y": np.array([0, 0, 1, 1], dtype=np.int8),
            "B": np.array([0, 1, 1, 1], dtype=np.int8),
            "D": np.array([1, 1, 0, 1], dtype=np.int8),
            "H": np.array([1, 0, 1, 1], dtype=np.int8),
            "V": np.array([1, 1, 1, 0], dtype=np.int8),
            "N": np.array([0, 0, 0, 0], dtype=np.int8)
        }
        self.virus_label_dictionary = self.create_labels_from_virus_list()
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
        print(label_dict)
        return label_dict

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
            fna_files_list = glob.glob(self.fasta_files_path + 'Viruses Raw Data/' + CurrentVirus + "/ncbi_dataset/data/GCA**/*.fna")
            # Iterate each file according to virus and extract segments into np array
            segment_per_virus = np.concatenate(list(map(self.segments_from_fna_file, fna_files_list)))
            # Create list of all np arrays by virus
            segments_virus_list.append(segment_per_virus)
        # This element contains array of arrays - by each virus- the whole segments
        self.set_all_segments_by_virus(segments_virus_list)

    def get_token_frags_and_labels(self):
        list_token_frags = []
        list_labels = []
        for virus_idx in range(self.viruses_num):
            tokened_frags = self.all_tokened_frags_by_virus[virus_idx]
            v_label = self.virus_label_dictionary[self.Viruses_list[virus_idx]]
            # v_path = self.create_single_path(self.Viruses_list[virus_idx])
            list_token_frags.append(tokened_frags)
            list_labels.append(v_label)
            # self.write_fragments_to_tfr(tokened_frags, v_label, v_path)
        return list_token_frags, list_labels

    # This is a complete method that use all the class abilities in order to create TFRecords for each labels.
    # Return: list_token_frags- list in viruses number len of all token frags for each virus
    #         list_labels- list in viruses number len of the label for each virus
    def generate_tokens_and_labels_from_scratch(self):
        self.segment_creation_from_fna()
        self.all_viruses_fragments()
        return self.get_token_frags_and_labels()

