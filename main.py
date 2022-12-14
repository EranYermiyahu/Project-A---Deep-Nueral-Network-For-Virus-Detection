import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import os
import os.path
import glob
from DNASeq import DNASeq
from DataSet import DataSet
from itertools import repeat


if __name__ == '__main__':
    dna_seq = DNASeq(["Test_Virus"])
    token_frags_list, labels_list = dna_seq.generate_tokens_and_labels_from_scratch()
    data_set = DataSet(dna_seq.Viruses_list)
    data_set.create_tfrecords(token_frags_list, labels_list)






    #
    # TFRecord_files_paths_list = dna_seq.generate_tfrecord_files_from_scratch()
    # filepath_dataset = tf.data.Dataset.list_files(self._getTrainPath() + "*")
    # # data_set = DataSet()
    # dna_seq.create_labels_from_virus_list()
    # dna_seq.segment_creation_from_fna()
    # dna_seq.all_viruses_fragments()
    # dna_seq.viruses_to_tfr_files()
    # #dna_seq.Crea
    #
    # tfrecord_path = '../TFRecords/Test_Virus.tfrecord'
    # #tensor = tf.io.read_file(tfrecord_file)
    # data_loader = tf.data.TFRecordDataset(tfrecord_path)
    # print(data_loader)
    #
    #
    #
    #
    #
