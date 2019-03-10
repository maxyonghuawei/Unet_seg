# Unet_seg

This is for the endoscopy vedio segmentation task with Unet and ConvLSTM structure.

Image is RGB image "uint8" from 0-255 converted to grey scale.

Mask is grey scale image with color_dict = [0, 50, 100, 150, 200] labeled 5 classes and 0 means the background.

The model is Unet embedded with ConvLSTM.
