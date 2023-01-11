import numpy as np
import tensorflow as tf
import os
import os.path
import glob
from DNASeq import DNASeq
from DataSet import DataSet
from LogisticRegression import LogisticRegression
from itertools import repeat


if __name__ == '__main__':
    dna_seq = DNASeq(virus_list=["Coronaviridae", "InfluenzaA", "Metapneumovirus", "Rhinovirus", "SarsCov2"]) #), fasta_files_path='../Viruses Raw Data/Test_Virus/')
    token_frags_list, labels_list = dna_seq.generate_tokens_and_labels_from_scratch()
    data_set = DataSet(dna_seq.Viruses_list)
    # data_set.create_tfrecords(token_frags_list, labels_list)

    # Create dataset of (tensor,label) from tfrecord for the whole data
    complete_data_set = data_set.create_dataset()
    # Split to test and train and configurate with features like epochs batches and shuffle keys.
    train_data_set, test_data_set = data_set.split_to_train_and_test_dataset(complete_data_set, epochs=30)
    # for x,y in test_data_set:
    #     print(y)
    model = LogisticRegression(input_shape=(4, 150), num_classes=data_set.viruses_num)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(train_data_set)
    # model.evaluate(test_data_set)
