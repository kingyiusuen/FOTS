# FOTS: Fast Oriented Text Spotting with a Unified Network

This is a Pytorch implementation of [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/pdf/1801.01671.pdf). It is **incomplete** at the moment. I haven't trained the model yet and am still fixing bugs.

## Model Architecture

The model largely draws inspiration from a general object detection algorithm. A CNN is first used to encode the input image into feature maps. Each spatial location at a feature map corresponds to a region of the input image. The feature maps are then passed into a text detection branch which consists of a classifier to predict the existence of text at each spatial location, and a regressor to predict the geometries of the bounding box (specifically, the distances to the four edges and the rotation angle). 

In a general objection detection task, there is a need for converting the proposed ROIs to fixed-size feature maps using operations like ROI pooling or ROI align, because the ROIs are of arbitrary size and they cannot be directly passed into a fully connected layer which requires a fixed input size. Similarly, we need to fit text not just of arbitrary sizes but also of arbitrary orientations. Therefore, the model uses an operation named ROI Rotate to rotate the proposed text regions (so that they are axis-aligned) and fix the height of the proposed regions. But in order to deal with the variations in text length, no restriction on the width is imposed. Since the transformed feature maps will be fed into a CRNN for text recognition instead of fully connected layers, arbitrary width is not a concern. 

After ROI rotate, the aforementioned CRNN will be used to predict the text labels. The CRNN consists of VGG-like sequential convolutions, max-poolings along the height-axis, a bi-directional LSTM as well as a fully-connection layer, which outputs softmax probabilities over the vocabulary. Finally, the outputs from different time steps are fed to a CTC decoder to get the raw text from the input image.

## How to Use

To train the model from scratch use the ``--train`` argument, followed by the name of the dataset you want to train with. For example,

```
$ python fots.py --train ICDAR2013
```

Supported datasets include [ICDAR2013](https://rrc.cvc.uab.es/?ch=2), [ICDAR2015](https://rrc.cvc.uab.es/?ch=4) and [SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/). The data directory, number of epochs, batch size and other configurations can be modified in ``config.yml``. By default, a checkpoint file named ``checkpoint.pt`` is saved every time the validation loss decreases. The checkpoint file stores the model parameters, the state of the optimizer and the state of the learning rate scheduler. The training will be terminated if the validation loss stops improving for a prespecified number of times. Also, the training and validation losses are logged in ``./logs/`` and can be visualized in Tensorboard using the following command:

```
tensorboard --logdir=logs
```

If you want to resume training from a checkpoint, add ``--ckpt`` with the path to the checkpoint file. The states of optimizer and learning rate scheduler from the checkpoint will overwrite the configurations in ``config.yml``. For example,

```
$ python fots.py --train ICDAR2013 --ckpt checkpoint.pt
```

To test the performance of a trained model, use ``--test``, followed by the name of the dataset and use ``--ckpt`` to specify the path to the trained model. The precision, recall and f1 score of the test dataset will be printed out. Note that SynthText does not have a test dataset. For example,

```
$ python fots.py --test ICDAR2013 --ckpt checkpoint.pt
```

## Pretrained Model
