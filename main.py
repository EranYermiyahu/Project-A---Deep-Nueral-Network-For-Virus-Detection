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
    data_set = DataSet(["Metapneumovirus"])
    # data_set = DataSet()
    data_set.segment_creation_from_fna()
    data_set.all_viruses_fragments()
    data_set.fragment_to_token(data_set.all_fragments_by_virus[0][0])





