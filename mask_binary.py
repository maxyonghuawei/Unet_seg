# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 22:14:02 2019

Turn the grey level Images to binary Images and save in the files

@author: Kumawife
"""
from __future__ import print_function
#import numpy as np
from PIL import Image
import os



#mask = np.ones([256,256,3])
#
#mask = mask[:,:,0]
#print(mask.shape)
#
#new_mask  = np.zeros(mask.shape + (3,))
#
#print(new_mask.shape)
#
#for i in range(3):
#    new_mask[mask==i, i] = 3
#
#print(new_mask)


#im = Image.open("frame049.png")
# format shows the image format, mode shows the image is "L grey scale" or "RGB"
#print(im.format, im.size, im.mode) 
#im.show()

#im_binary = im.convert("1")
#im_binary.show()
#print(im_binary.size, im_binary.mode)

def grey_to_binary(img, thresh):
    """
    Converting the grey scale image to binary image through the PIL API
    defining the "thresh" for binarize the pixel in the input image
    """
    thresh = thresh
    fn = lambda x: 255 if x>thresh else 0
    r = img.point(fn, mode="1")
#    r.save('foo.png')
    print("convert mode from grey to binary")
    return r

def saveImage(in_path, out_path, thresh):
    """
    Enter the grey images input path "in_path"
    Enter the save path "out_path"
    The threshold for the frey scale images transfered to binary images
    """
    if os.path.exists(in_path):
        for img in os.listdir(in_path):
            temp = in_path + '/' + img
            img_pic = Image.open(temp)
            img_bir = grey_to_binary(img_pic, thresh)
            f, e = os.path.splitext(img)
            outname = f + ".jpg"
            final_out_path = out_path + outname
#            print(final_out_path)
            try:
                img_bir.save(final_out_path)
            except IOError:
                print("Fail to save the file")
        print("Binary pictures are successfully saved")
    else:
        print("Input path is not exist.")
            

#im_bir_test = grey_to_binary(im, 25)
#im_bir_test.show()
        

if __name__ == "__main__":

    out_path = 'D:/Msc.project/data_2017/training/instrument_dataset_1/instrument_dataset_1/ground_truth/other/'
    in_path = 'D:/Msc.project/data_2017/training/instrument_dataset_1/instrument_dataset_1/ground_truth/Other_labels'
    thresh = 7
    saveImage(in_path, out_path, thresh)
