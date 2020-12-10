# FOTS: Fast Oriented Text Spotting with a Unified Network

## Introduction

This model largely draws inspiration from a general object detection algorithm. A CNN is first used to encode the input image into feature maps. Each spatial location at a feature map corresponds to a region of the input image. The feature maps are then passed into a text detection branch which consists of a classifier to predict the existence of text at each spatial location, and a regressor to predict the geometries of the bounding box (specifically, the distances to the four edges and the rotation angle). 

In a general objection detection task, there is a need for converting the proposed ROIs to fixed-size feature maps using operations like ROI pooling or ROI align, because the ROIs are of arbitrary size and they cannot be directly passed into a fully connected layer which requires a fixed input size. Similarly, we need to fit text not just of arbitrary sizes but also of arbitrary orientations. Therefore, the model uses an operation named ROI Rotate to rotate the proposed text regions (so that they are axis-aligned) and fix the height of the proposed regions. But in order to deal with the variations in text length, no restriction on the width is imposed. Since the transformed feature maps will be fed into a CRNN for text recognition instead of fully connected layers, arbitrary width is not a concern. 

After ROI rotate, the aforementioned CRNN will be used to predict the text labels. The CRNN consists of VGG-like sequential convolutions, max-poolings along the height-axis, a bi-directional LSTM as well as a fully-connection layer, which outputs softmax probabilities over the vocabulary. Finally, the outputs from different time steps are fed to a CTC decoder to get the raw text from the input image.