# This is a sample Python script.

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import os
import os.path
import glob
from itertools import repeat


# Need to implement as class
class DataSet:
    def __init__(self, virus_list=["Coronaviridae", "InfluenzaA", "Metapneumovirus", "Rhinovirus", "SarsCov2"], fragment_size=150):
        self.Viruses_list = virus_list
        self.viruses_num = len(virus_list)
        self.fragment_size = fragment_size
        self.nuc_translation_dictionary = {
            "A": np.array([1, 0, 0, 0], dtype=np.int8),
            "G": np.array([0, 1, 0, 0], dtype=np.int8),
            "C": np.array([0, 0, 1, 0], dtype=np.int8),
            "T": np.array([0, 0, 0, 1], dtype=np.int8),
            "N": np.array([0, 0, 0, 0], dtype=np.int8)
        }
        # This element contains array of arrays - by each virus- the whole segments
        self.all_segments_by_virus = None
        self.all_fragments_by_virus = None

    def set_all_segments_by_virus(self, seg_by_vir_list):
        self.all_segments_by_virus = np.asarray(seg_by_vir_list)

    def set_all_fragments_by_virus(self, frag_by_vir_list):
        self.all_fragments_by_virus = np.asarray(frag_by_vir_list)

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
        # for each segment create list of the valid indexes
        interval = len(seg) - self.fragment_size
        valid_index_per_seg_list = list(range(interval))
        # For each segment change the list of start indexes to numpy of fragments
        all_fragments_per_seg = np.asarray(list(map(self.start_idx_to_fragment, valid_index_per_seg_list, repeat(seg))))
        return all_fragments_per_seg

    # Return matrix num_of_frag X len(segments) with randomly fragments from the segments
    def generate_fragmnets_from_segments(self, segments):
        # all_fragments_per_virus = np.array([])
        #SiF = np.vectorize(self.start_idx_to_fragment)
        all_fragments_per_virus = np.concatenate(list(map(self.one_segment_to_fragments, segments)), axis=0)
        print(len(all_fragments_per_virus))

        # for seg in segments:
        #     # for each segment create list of the valid indexes
        #     valid_index_per_seg_list = list(range(len(seg)-self.fragment_size))
        #     #valid_index_per_seg = np.arange(0, len(seg)-fragment_size, 1)
        #     # For each segment change the list of start indexes to numpy of fragments
        #     all_fragments_per_segments = np.asarray(list(map(self.start_idx_to_fragment, valid_index_per_seg_list, seg)))
        #     #all_fragments_per_segments = SiF(valid_index_per_seg, seg, fragment_size)
        #     all_fragments_per_virus = np.append(all_fragments_per_virus, all_fragments_per_segments)
        return all_fragments_per_virus

    def all_viruses_fragments(self):
        all_fragments_by_virus_list = list(map(self.generate_fragmnets_from_segments, self.all_segments_by_virus))
        self.set_all_fragments_by_virus(all_fragments_by_virus_list)
        print(self.all_fragments_by_virus[0])
        print("hi")
        # for j in range(self.viruses_num):
        #     list_fragments_by_virus.append(self.generate_fragmnets_from_segment(self.all_segments_by_virus[j]))
        # all_fragments_by_virus = np.asarray(list_fragments_by_virus)
        # fragment_to_token(all_fragments_by_virus[0][0])

    def char_to_vector(self, char):
        return self.nuc_translation_dictionary[char]

    def fragment_to_token(self, fragment):
        # Create np array of chars, each char is nucleotide
        broken_frag = np.asarray([char for char in fragment])
        # Create fragsize X 4 matrix from each nucleotide, using map function fo convertion
        frag_Tensor = np.asarray(list(map(self.char_to_vector, broken_frag.T)))
        print(frag_Tensor.shape)
        return frag_Tensor

    def segment_creation_from_fna(self):
        segments_virus_list = []
        # Iterate each virus and create np array of segments
        for CurrentVirus in self.Viruses_list:
            # List of all fna name files per virus
            fna_files_list = glob.glob("../Viruses Raw Data/{}/ncbi_dataset/data/GCA**/*.fna".format(CurrentVirus))
            #segment_per_virus = np.empty(0, dtype=np.str)
            # Iterate each file according to virus and extract segments into np array
            segment_per_virus = np.concatenate(list(map(self.segments_from_fna_file, fna_files_list)))
            # Create list of all np arrays by virus
            segments_virus_list.append(segment_per_virus)
        # This element contains array of arrays - by each virus- the whole segments
        self.set_all_segments_by_virus(segments_virus_list)




# Press the green button in the gutter to run the script
if __name__ == '__main__':
    # data_set = DataSet(["Coronaviridae"])
    data_set = DataSet()
    data_set.segment_creation_from_fna()
    print(len(data_set.all_segments_by_virus[4]))
    data_set.all_viruses_fragments()
    # # Viruses_list = ["Coronaviridae", "InfluenzaA", "Metapneumovirus", "Rhinovirus", "SarsCov2"]
    # Viruses_list = ["Metapneumovirus"]
    # viruses_num = len(Viruses_list)
    # segments_virus_list = []
    # # Iterate each virus and create np array of segments
    # for CurrentVirus in Viruses_list:
    #     # List of all fna name files per virus
    #     fna_files_list = glob.glob("../Viruses Raw Data/{}/ncbi_dataset/data/GCA**/*.fna".format(CurrentVirus))
    #     segment_per_virus = np.empty(0, dtype=np.str)
    #     # Iterate each file according to virus and extract segments into np array
    #     for fna_file_name in fna_files_list:
    #         cur_file = open(fna_file_name, 'r')
    #         raw_text = cur_file.read()
    #         # Parse raw data into np of segments
    #         raw_segment_list = raw_text.split('>')
    #         raw_segments = (np.asarray(raw_segment_list))[1:]
    #         raw_segments = np.char.partition(raw_segments, '\n')[:, 2]
    #         segments_per_file = np.char.replace(raw_segments, '\n', '')
    #         segment_per_virus = np.append(segment_per_virus, segments_per_file)
    #     # Create list of all np arrays by virus
    #     segments_virus_list.append(segment_per_virus)
    # # This element contains array of arrays - by each virus- the whole segments
    # all_segments_by_virus = np.asarray(segments_virus_list)
    print('NOT RELEVANT')
    list_fragments_by_virus = []
    for j in range(viruses_num):
        list_fragments_by_virus.append(generate_fragmnets_from_segment(all_segments_by_virus[j]))
    all_fragments_by_virus = np.asarray(list_fragments_by_virus)
    fragment_to_token(all_fragments_by_virus[0][0])





