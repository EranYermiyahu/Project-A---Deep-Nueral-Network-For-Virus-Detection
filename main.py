# This is a sample Python script.

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import os
import os.path
import glob

def start_idx_to_fragment(num, seg, fragment_size):
    ret = seg[num:num + (fragment_size-1)]
    return ret

def generate_fragmnets_from_segment(segments, num_of_frag, fragment_size=150):
    num_of_segments = len(segments)
    max_idx_per_segment = np.char.str_len(segments)-fragment_size
    samples_fragmnent_matrix = np.chararray((num_of_frag, num_of_segments), itemsize=fragment_size)
    random_integers = np.random.randint(0, max_idx_per_segment, size=(num_of_frag, num_of_segments))
    print(random_integers)
    SiF = np.vectorize(start_idx_to_fragment)
    for i in range(num_of_segments):
        samples_fragmnent_matrix[:, i] = SiF(random_integers[:, i], segments[i], fragment_size)
    print(samples_fragmnent_matrix)


# Press the green button in the gutter to run the script
if __name__ == '__main__':
    Viruses_list = ["Coronaviridae", "InfluenzaA", "Metapneumovirus", "Rhinovirus", "SarsCov2"]
    segments_virus_list = []
    # Iterate each virus and create np array of segments
    for CurrentVirus in Viruses_list:
        # List of all fna name files per virus
        fna_files_list = glob.glob("../Viruses Raw Data/{}/ncbi_dataset/data/GCA**/*.fna".format(CurrentVirus))
        segment_per_virus = np.empty(0, dtype=np.str)
        # Iterate each file according to virus and extract segments into np array
        for fna_file_name in fna_files_list:
            cur_file = open(fna_file_name, 'r')
            raw_text = cur_file.read()
            # Parse raw data into np of segments
            raw_segment_list = raw_text.split('>')
            raw_segments = (np.asarray(raw_segment_list))[1:]
            raw_segments = np.char.partition(raw_segments, '\n')[:, 2]
            segments_per_file = np.char.replace(raw_segments, '\n', '')
            segment_per_virus = np.append(segment_per_virus, segments_per_file)
        # Create list of all np arrays by virus
        segments_virus_list.append(segment_per_virus)
    # This element contains array of arrays - by each virus- the whole segments
    all_segments_by_virus = np.asarray(segments_virus_list)

    generate_fragmnets_from_segment(all_segments_by_virus[2], 10)





