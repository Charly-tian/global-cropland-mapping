import os
import time
from collections import OrderedDict
from datetime import datetime
import pandas as pd
from keras import backend as K
from keras.callbacks import Callback


class LogCallback(Callback):
    """ record training logs, including training / validation loss, metrics, iterval """

    def __init__(self, log_dir='./logs', log_id=None):
        super(LogCallback, self).__init__()
        self.log_dir = log_dir
        self.log_id = log_id
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if self.log_id is None:
            self.log_id = 'log_{}'.format(
                datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.log_path = os.path.join(log_dir, self.log_id + '.txt')

        self.run_start_time = None
        self.epoch_start_time = None
        self.run_data = []

    def on_train_begin(self, logs=None):
        self.run_start_time = time.time()

    def on_train_end(self, logs=None):
        self.run_data = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = (time.time() - self.epoch_start_time) / 60.
        run_duration = (time.time() - self.run_start_time) / 60.

        res = OrderedDict()
        res['epoch'] = epoch + 1
        for k in self.params['metrics']:
            res[k] = logs[k]
        res['lr'] = K.get_value(self.model.optimizer.lr)
        # res['batch_size'] = logs['size']
        res['epoch_duration'] = epoch_duration
        res['run_duration'] = run_duration
        self.run_data.append(res)
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(
            self.log_path, index=False)
