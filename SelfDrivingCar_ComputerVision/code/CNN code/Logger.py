import os
import json
import tensorflow as tf

class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, dict_name, log_file='training_log.json'):
        super().__init__()
        self.dict_name = dict_name
        self.log_file = log_file
        self.history = {}

        print(f'\nRecording metrics in {log_file} -> {dict_name}\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if "epoch" not in self.history:
            self.history["epoch"] = []
        self.history["epoch"].append(epoch + 1)

        for metric, value in logs.items():
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(value)


        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                full_log = json.load(f)
        else:
            full_log = {}

        full_log[self.dict_name] = self.history

        with open(self.log_file, 'w') as f:
            json.dump(full_log, f, indent=4)