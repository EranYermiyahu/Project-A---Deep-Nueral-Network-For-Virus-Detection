import numpy as np
import tensorflow as tf
from datetime import datetime
from DNASeq import DNASeq
from DataSet import DataSet
from LogisticRegression import LogisticRegression
from CNN import CNN
from Log_Folder import LogFolder
import matplotlib.pyplot as plt
import pickle
from itertools import repeat

EPOCHS = 100
BATCH_SIZE = 4096

def get_train_test_dataset(virus_list, create_tfr_files=False):
    dna_seq = DNASeq(virus_list=virus_list)
    data_set = DataSet(dna_seq.Viruses_list)
    if create_tfr_files:
        token_frags_list, labels_list = dna_seq.generate_tokens_and_labels_from_scratch()
        data_set.create_tfrecords(token_frags_list, labels_list)

    print("========= Created TFRecords - start to build Dataset =========")
    train_data_set = data_set.create_train_dataset(train_batch_size=BATCH_SIZE)
    test_data_set = data_set.create_test_dataset()
    print("========= Finished DataSet Creation - Define model and train it =========")
    return dna_seq, (train_data_set, test_data_set)


def train_model(model, train_data, optimizer='adam', loss='categorical_crossentropy'):
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    fit = model.fit(train_data, epochs=EPOCHS)
    return fit


def check_gpu():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


if __name__ == '__main__':
    today = datetime.now()
    check_gpu()
    virus_class_list = ["Coronaviridae", "InfluenzaA", "Rhinovirus", "SarsCov2"]#, "Adenovirus", "RSV"]
    dna_seq, dataset_tuple = get_train_test_dataset(virus_class_list, create_tfr_files=True)

    train_dataset = dataset_tuple[0]
    test_dataset = dataset_tuple[1]
    input_dim = (dna_seq.fragment_size, 4, 1)

    # Define Model
    # model = CNN(input_shape=input_dim, num_classes=dna_seq.viruses_num, name='CNN_3_layers')
    model = LogisticRegression(input_shape=input_dim, num_classes=dna_seq.viruses_num)

    fit = train_model(model, train_dataset)
    evaluation_results = model.evaluate(test_dataset)

    # Start to save the results
    # Save Checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    checkpoint_name = f'../Checkpoints/model_{model.model_name}_{dna_seq.viruses_num}viruses' + today.strftime('%d_%m_%Y-%H_%M')
    ckpt.save(file_prefix=checkpoint_name)
    history_name = f'../History/model_{model.model_name}_{dna_seq.viruses_num}viruses' + today.strftime('%d_%m_%Y-%H_%M') + '.pickle'
    with open(history_name, 'wb') as file:
        pickle.dump(fit.history, file)
    # ckpt.restore(checkpoint_name)

    # Distinguish between TFR creation scenario and Ready TFR scenario
    if dna_seq.generated_TFR:
        len_list = [len(frags) for frags in dna_seq.all_tokened_frags_by_virus]
    else:
        len_list = None
    log_folder = LogFolder(time=today, checkpoint_path=checkpoint_name + "-1.index",
                           tfr_path='../TFRecords', virus_list=dna_seq.Viruses_list,
                           len_list=len_list,
                           model_name=model.model_name, train_accuracy=fit.history['accuracy'][-1],
                           train_loss=fit.history['loss'][-1],
                           test_loss=evaluation_results[0], test_accuracy=evaluation_results[1])

    train_loss = fit.history['loss']
    # val_loss = fit.history['val_loss']

    plt.plot(range(1, EPOCHS + 1), train_loss, label='Training Loss')
    # plt.plot(range(1, EPOCHS + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()













    # today = datetime.now()
    # # dna_seq = DNASeq(virus_list=["Coronaviridae", "InfluenzaA", "Metapneumovirus", "Rhinovirus", "SarsCov2"])
    # dna_seq = DNASeq(virus_list=["Coronaviridae", "InfluenzaA", "Rhinovirus", "SarsCov2", "Adenovirus", "RSV"])
    # # dna_seq = DNASeq(virus_list=["Coronaviridae", "SarsCov2"])
    # # dna_seq = DNASeq(virus_list=["InfluenzaA", "Metapneumovirus", "Rhinovirus"])
    # # token_frags_list, labels_list = dna_seq.generate_tokens_and_labels_from_scratch()
    # data_set = DataSet(dna_seq.Viruses_list)
    # # data_set.create_tfrecords(token_frags_list, labels_list)
    # # data_set.print_labels_histogram(train_data_set, dna_seq.virus_label_dictionary)
    #
    #
    # # Create model and train it
    # print("========= Created TFRecords - start to build Dataset =========")
    # if tf.test.is_gpu_available():
    #     device = '/GPU:0'  # Use the first GPU device
    # else:
    #     device = '/CPU:0'  # Use the CPU device
    #
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #
    # with tf.device(device):
    #     # Create dataset of (tensor,label) from tfrecord of the training and test
    #     # and config features like epochs batches and shuffle keys.
    #     train_data_set = data_set.create_train_dataset()
    #     test_data_set = data_set.create_test_dataset()
    #     print("========= Finished DataSet Creation - Define model and train it =========")
    #     # model = LogisticRegression(input_shape=(4, 150), num_classes=data_set.viruses_num)
    #     model = CNN(input_shape=(150, 4, 1), num_classes=data_set.viruses_num, name='CNN_3_layers')
    #     model.compile(optimizer='adam', loss='categorical_crossentropy',
    #                   metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    #     fit = model.fit(train_data_set)
    #     evaluation_results = model.evaluate(test_data_set)
    #
    #     # Save Checkpoint
    #     ckpt = tf.train.Checkpoint(model=model)
    #     checkpoint_name = f'../Checkpoints/model_{model.model_name}_{dna_seq.viruses_num}viruses' + today.strftime('%d_%m_%Y-%H_%M')
    #     ckpt.save(file_prefix=checkpoint_name)
    #     history_name = f'../History/model_{model.model_name}_{dna_seq.viruses_num}viruses' + today.strftime('%d_%m_%Y-%H_%M') + '.pickle'
    #     with open(history_name, 'wb') as file:
    #         pickle.dump(fit.history, file)
    #     # ckpt.restore(checkpoint_name)
    #
    #     # Distinguish between TFR creation scenario and Ready TFR scenario
    #     if dna_seq.generated_TFR:
    #         len_list = [len(frags) for frags in dna_seq.all_tokened_frags_by_virus]
    #     else:
    #         len_list = None
    #     log_folder = LogFolder(time=today, checkpoint_path=checkpoint_name + "-1.index", tfr_path='../TFRecords', virus_list=dna_seq.Viruses_list,
    #                            len_list=len_list,
    #                            model_name=model.model_name, train_accuracy=fit.history['accuracy'][-1],
    #                            train_loss=fit.history['loss'][-1], test_loss=test_loss, test_accuracy=test_accuracy)
    #
    # # Plot the loss vs. epochs
    # plt.plot(range(1, 500 + 1), fit.history['loss'])
    # plt.title('Loss vs. Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Train Loss')
    # plt.show()
    #
    # #     pred_virus_idx = tf.math.argmax(ret_val, axis=1)
    # #     print(f"predicted virus in index {pred_virus_idx} , \
    # #     However, label is {tf.math.argmax(labels, axis=1)}")
    # #
    # #     break

