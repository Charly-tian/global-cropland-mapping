"""
    This script is the configures for building & debugging & fine-tunning models on my local machine.
"""

import os
import numpy as np
from keras import optimizers
from keras import callbacks
from semantic_segmentation.utils.train.losses import get_loss

# ==================== Data ====================
# # # Data process
channelMin_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -400])
channelMax_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0,  1.0, 2000])

# # # Train & Val Dataset
workEnv_ = './runs/C2_test'
dataEnv_ = './dataset/C2_test'
dataDir_ = dataEnv_ + '/data'
trainFileText_ = dataEnv_ + '/train.txt'
valFileText_ = dataEnv_ + '/val.txt'
batchSize_ = 1

# # # Model Parameters
modelFlag_ = "sbce_pspnet"
backboneName_ = "resnet50"
backboneWeights_ = "D:/keras_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
inputHeight_ = 192
inputWidth_ = 192
inputChannel_ = 7
numClass_ = 2
labelProbToCls_ = False
oneHot_ = True
numFilters_ = 32

# # # Training Parameters
lossName_ = "jl"
metricNames_ = ['jc']
optimizerName_ = "adam"
learningRate_ = 3e-4
epoch_ = 200
plateauPatience_ = 15
verbose_ = 1
debug_ = True
modelSummary_ = True
resumeVersion_ = ""
saveVersion_ = "{}_{}x{}x{}".format(
    modelFlag_, inputHeight_, inputWidth_, inputChannel_)

# # # below are not editable
loss_ = get_loss(lossName_)
metrics_ = [get_loss(_) for _ in metricNames_]
optimizer_ = optimizers.Adam(lr=learningRate_)

checkpointDir_ = os.path.join(workEnv_, 'checkpoints')
logsDir_ = os.path.join(workEnv_, 'logs')
plotsDir_ = os.path.join(workEnv_, 'plots')
os.makedirs(checkpointDir_, exist_ok=True)
os.makedirs(logsDir_, exist_ok=True)
os.makedirs(plotsDir_, exist_ok=True)
resumeModelPath_ = os.path.join(checkpointDir_, resumeVersion_ + '.h5')
saveModelPath_ = os.path.join(checkpointDir_, saveVersion_ + '.h5')
plotPath_ = os.path.join(plotsDir_, saveVersion_ + '.png')


callbacks_ = [
    callbacks.ModelCheckpoint(monitor='val_dc', filepath=saveModelPath_, mode='max',
                              save_best_only=True, save_weights_only=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=plateauPatience_,
                                min_delta=5e-3, min_lr=1e-6, verbose=1),
    callbacks.CSVLogger(filename=os.path.join(logsDir_, saveVersion_ + '.csv'))
]
