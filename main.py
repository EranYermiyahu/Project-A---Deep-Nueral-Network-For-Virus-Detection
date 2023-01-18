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
    # I Love You
    dna_seq = DNASeq(virus_list=["Coronaviridae", "InfluenzaA", "Metapneumovirus", "Rhinovirus", "SarsCov2"])
    token_frags_list, labels_list = dna_seq.generate_tokens_and_labels_from_scratch()
    data_set = DataSet(dna_seq.Viruses_list)
    data_set.create_tfrecords(token_frags_list, labels_list)

    print("========= Created TFRecords - start to build Dataset =========")
    # Create dataset of (tensor,label) from tfrecord of the training and test anf config features like epochs batches and shuffle keys.
    train_data_set = data_set.create_train_dataset()
    test_data_set = data_set.create_test_dataset()
    # Create model and train it
    model = LogisticRegression(input_shape=(4, 150), num_classes=data_set.viruses_num)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data_set)
    model.evaluate(test_data_set)
