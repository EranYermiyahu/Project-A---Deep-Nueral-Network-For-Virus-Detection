import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import os
import os.path
import glob
from Data_Set import DataSet
from itertools import repeat


if __name__ == '__main__':
    # data_set = DataSet(["Test_Virus"])
    data_set = DataSet()
    data_set.create_labels_from_virus_list()
    data_set.segment_creation_from_fna()
    data_set.all_viruses_fragments()
    data_set.viruses_to_tfr_files()





