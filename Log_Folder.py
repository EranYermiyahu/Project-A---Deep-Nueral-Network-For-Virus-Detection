from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import os
import os.path
import shutil
import glob


class LogFolder:
    def __init__(self, checkpoint_path, tfr_path, virus_list, len_list, model_name,
                 train_accuracy, test_accuracy, train_loss, test_loss, path='../log_directory/'):
        if os.path.exists(path) is False:
            os.mkdir(path)
        # Create the relevant directories
        today = datetime.now()
        self.location_path = path + today.strftime('%d_%m_%Y-%H_%M')
        os.mkdir(self.location_path)

        with open(self.location_path + '/simulation_details.txt', 'w') as file:
            data_to_insert = f"Simulation executed on {today.strftime('%Y%m%d%H%M')} details : \n" \
                             f"Viruses Learned are: {len(virus_list)} \n"
            for idx, virus in enumerate(virus_list):
                virus_detail = f"{virus} : {len_list[idx]} Tokens \n"
                data_to_insert += virus_detail

            model_detail = f"\n\nThe Training Model Is: {model_name}\n" \
                           f"Training Accuracy: {train_accuracy} and Loss: {train_loss}\n " \
                           f"Testing Accuracy: {test_accuracy} and Loss: {test_loss}\n "

            data_to_insert += model_detail

            file.write(data_to_insert)

        shutil.copytree(tfr_path, self.location_path + "/TFRecords")
        shutil.copy2(checkpoint_path, self.location_path + "/checkpoint")





