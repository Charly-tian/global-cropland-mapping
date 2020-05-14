from keras.utils.vis_utils import plot_model
from keras import backend as K
from cfgs import *
import tensorflow as tf

from semantic_segmentation.nn.models.deeplab_v3p import deeplabv3p
from semantic_segmentation.nn.models.unet import unet_bn, resunet
from semantic_segmentation.nn.models.pspnet import pspnet
from semantic_segmentation.nn.models.refinenet import refinenet_4cascaded, refinenet_1cascaded
from semantic_segmentation.nn.models.srinet import sri_net
from semantic_segmentation.utils import log, plot_image_label_per_channel
from semantic_segmentation.utils.data import dataset, datagenerator, transformer


def train():
    train_dataset = dataset.H5SegmentationDataset(
        root=dataDir_, fn_txt=trainFileText_,
        n_class=numClass_, label_prob_to_cls=labelProbToCls_,
        label_one_hot=oneHot_,
        transforms=transformer.Compose([
            transformer.MinMaxScale(mins=channelMin_, maxs=channelMax_),
            transformer.RandomGammaTransform(gamma=0.5, p=0.5),
            transformer.RandomBlur(kernel_size=(3, 3), p=0.1),
            transformer.RandomHorizontalFlip(p=0.5),
            transformer.RandomVerticalFlip(p=0.5),
            transformer.RandomRotate(angle_threshold=20, p=0.5),
            transformer.RandomScale(scale_range=(0.8, 1.2), p=0.5),
            # transformer.Normalize(mean=channelMean_, std=channelStd_)
        ])
    )
    val_dataset = dataset.H5SegmentationDataset(
        root=dataDir_, fn_txt=valFileText_,
        n_class=numClass_, label_prob_to_cls=labelProbToCls_,
        label_one_hot=oneHot_,
        transforms=transformer.Compose([
            # transformer.Normalize(mean=channelMean_, std=channelStd_)
            transformer.MinMaxScale(mins=channelMin_, maxs=channelMax_)
        ])
    )

    train_generator = datagenerator.data_generator(
        train_dataset, batch_size=batchSize_, shuffle=True)
    val_generator = datagenerator.data_generator(
        val_dataset, batch_size=batchSize_, shuffle=False)
    stepsPerValEpoch_ = len(val_dataset) // batchSize_
    stepsPerEpoch_ = stepsPerValEpoch_ * 5

    # for debug
    if debug_:
        for _ in range(1):
            x, y = next(train_generator)
            print(x.shape, y.shape)
            if oneHot_:
                plot_image_label_per_channel(x[0], np.argmax(y[0], -1))
            else:
                from semantic_segmentation.utils import plot_rgb_image
                plot_image_label_per_channel(x[0], y[0, :, :, 0])
                plot_rgb_image(x[0], y[0, :, :, 0])

    # define model, load weights, compile model
    model = pspnet(input_shape=(inputHeight_, inputWidth_, inputChannel_),
                   n_class=numClass_, one_hot=oneHot_, backbone_name=backboneName_,
                   backbone_weights=backboneWeights_, output_stride=8)

    if os.path.exists(resumeModelPath_):
        log('load weights from `{}`'.format(resumeModelPath_))
        model.load_weights(resumeModelPath_, by_name=True, skip_mismatch=True)
    else:
        log('build new model `{}`'.format(saveModelPath_))
        plot_model(model, to_file=plotPath_)
    with open(plotPath_.replace('png', 'json'), 'w') as f:
        f.write(model.to_json())
    if modelSummary_:
        model.summary()
        log('model input: {}, output: {}'.format(
            model.input_shape, model.output_shape))
        log('trainable parameters: {}'.format(
            int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))))
        log('non-trainable parameters: {}'.format(
            int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))))
    model.compile(optimizer=optimizer_, loss=loss_, metrics=['acc'] + metrics_)

    # train
    model.fit_generator(
        generator=train_generator, steps_per_epoch=stepsPerEpoch_,
        validation_data=val_generator, validation_steps=stepsPerValEpoch_,
        epochs=epoch_, callbacks=callbacks_, verbose=verbose_, initial_epoch=0)
    model.save_weights(filepath=checkpointDir_ +
                       '/{}_{}.h5'.format(saveVersion_, epoch_))
    log('training success!')


if __name__ == '__main__':
    print(os.getcwd())
    train()
