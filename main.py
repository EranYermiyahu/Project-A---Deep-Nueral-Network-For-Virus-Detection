import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import os.path
import glob
from DNASeq import DNASeq
from DataSet import DataSet
from LogisticRegression import LogisticRegression
from CNN import CNN
from Log_Folder import LogFolder
import matplotlib.pyplot as plt
from itertools import repeat


if __name__ == '__main__':

    today = datetime.now()
    # dna_seq = DNASeq(virus_list=["Coronaviridae", "InfluenzaA", "Metapneumovirus", "Rhinovirus", "SarsCov2"])
    dna_seq = DNASeq(virus_list=["Coronaviridae", "InfluenzaA", "Rhinovirus", "SarsCov2", "Adenovirus", "RSV"])
    # dna_seq = DNASeq(virus_list=["Coronaviridae", "SarsCov2"])
    # dna_seq = DNASeq(virus_list=["InfluenzaA", "Metapneumovirus", "Rhinovirus"])
    # token_frags_list, labels_list = dna_seq.generate_tokens_and_labels_from_scratch()
    data_set = DataSet(dna_seq.Viruses_list)
    # data_set.create_tfrecords(token_frags_list, labels_list)

    # data_set.print_labels_histogram(train_data_set, dna_seq.virus_label_dictionary)

    # Create model and train it
    print("========= Created TFRecords - start to build Dataset =========")
    # model = LogisticRegression(input_shape=(4, 150), num_classes=data_set.viruses_num)
    if tf.test.is_gpu_available():
        device = '/GPU:0'  # Use the first GPU device

    else:
        device = '/CPU:0'  # Use the CPU device
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


    with tf.device(device):
        # Create dataset of (tensor,label) from tfrecord of the training and test
        # and config features like epochs batches and shuffle keys.
        train_data_set = data_set.create_train_dataset()
        test_data_set = data_set.create_test_dataset()
        print("========= Finished DataSet Creation - Define model and train it =========")
        model = CNN(input_shape=(150, 4, 1), num_classes=data_set.viruses_num, name='CNN_3_layers')
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        # train_data_set = tf.expand_dims(train_data_set, axis=-1)
        # train_data_set = tf.repeat(train_data_set, 3, axis=-1)
        #
        # test_data_set = tf.expand_dims(test_data_set, axis=-1)
        # test_data_set = tf.repeat(test_data_set, 3, axis=-1)
        fit = model.fit(train_data_set)
        test_loss, test_accuracy = model.evaluate(test_data_set)

        # Save Checkpoint
        ckpt = tf.train.Checkpoint(model=model)
        checkpoint_name = '../Checkpoints/' + today.strftime('%d_%m_%Y-%H_%M')
        ckpt.save(file_prefix=checkpoint_name)
        # ckpt.restore(checkpoint_name)
        log_folder = LogFolder(time=today, checkpoint_path=checkpoint_name + "-1.index", tfr_path='../TFRecords', virus_list=dna_seq.Viruses_list,
                               len_list=[len(frags) for frags in dna_seq.all_tokened_frags_by_virus],
                               model_name=model.model_name, train_accuracy=fit.history['accuracy'][-1],
                               train_loss=fit.history['loss'][-1], test_loss=test_loss, test_accuracy=test_accuracy)
    #
    # for tokens, labels in test_data_set:
    #     ret_val = model.call(tokens)

    # Plot the loss vs. epochs
    plt.plot(range(1, 500 + 1), fit.history['loss'])
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.show()

    #     pred_virus_idx = tf.math.argmax(ret_val, axis=1)
    #     print(f"predicted virus in index {pred_virus_idx} , \
    #     However, label is {tf.math.argmax(labels, axis=1)}")
    #
    #     break

