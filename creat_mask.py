# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 18:00:12 2019

@author: Kumawife
"""
import numpy as np
from PIL import Image
import os
import cv2
from pylab import plt

#def transparent_mask(image, num_class):
#    """
#    input image "img"
#    number of classes num_class you want
#    """
#    img = Image.open(image)
#    img = np.array(img) # transfer to numpy array
#    transpare_mask = np.zeros(img.shape+(num_class,))
#    
#    return transpare_mask


#img = transparent_mask("frame049.png", 4)
#img = np.array(Image.open("frame049.png"))
#print(img.shape, img.dtype)

#Sky = [128,128,128]
#Building = [128,0,0]
#Pole = [192,192,128]
#Road = [128,64,128]
#Pavement = [60,40,222]
#Tree = [128,128,0]
#SignSymbol = [192,128,128]
#Fence = [64,64,128]
#Car = [64,0,128]
#Pedestrian = [64,64,0]
#Bicyclist = [0,128,192]
#Unlabelled = [0,0,0]

#color_dict = np.array([Sky, Building, Pole, Road, Pavement,
#                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


color_dict = [50, 100, 150, 200] #defining the grey scale pixels for
# Left_labels, Maryland_labels, Ot_labels, Right_labels seperatelly

def pro_adjust_data(path_list, num_class, save_path):
    
#    temp_path = path_list[0] + '/' + 'frame000.jpg'
#    with open(temp_path, 'rb') as t:
#        mask_shape = np.array(Image.open(t))
#        trans_mask = np.zerps(mask_shape.shape)
    if os.path.exists(path_list[0][0]):
        for image in os.listdir(path_list[0][0]):
            
#            temp_path = path_list[0][0] + '/' + image
#            mask_shape = np.array(Image.open(temp_path))
            trans_mask = np.zeros(shape=[1080,1920])
            
            
#            temp0 = path_list[0][0] + '/' + image
#            temp1 = path_list[1][0] + '/' + image
#            temp2 = path_list[2][0] + '/' + image
#            temp3 = path_list[3][0] + '/' + image
            
#            mask0 = np.array(Image.open(temp0))
#            mask1 = np.array(Image.open(temp1))
#            mask2 = np.array(Image.open(temp2))
#            mask3 = np.array(Image.open(temp3))
            
            for i in range(num_class):
                temp = path_list[i][0] + '/' + image
                mask = np.array(Image.open(temp))
                trans_mask[mask==255] = color_dict[i]
                
            plt.imshow(trans_mask)
            
            f, e = os.path.splitext(image)
            save_dir = save_path + '/' + f + '.jpg'
            try:
#                plt.imshow(trans_mask)
                cv2.imwrite(save_dir, trans_mask)
            except IOError:
                print("Fail to save the mask image to file")
    else:
        print("Input path is not exist.")
            
        
if __name__ == "__main__":
    Ot_labels = ['D:/Msc.project/data_2017/training/instrument_dataset_1/instrument_dataset_1/ground_truth/other']
    Left_labels = ['D:/Msc.project/data_2017/training/instrument_dataset_1/instrument_dataset_1/ground_truth/left']
    Right_labels = ['D:/Msc.project/data_2017/training/instrument_dataset_1/instrument_dataset_1/ground_truth/right']
    Maryland_labels = ['D:/Msc.project/data_2017/training/instrument_dataset_1/instrument_dataset_1/ground_truth/maryland']
    
    path_list = [Left_labels, Maryland_labels, Ot_labels, Right_labels]
    num_class = 4
    save_path = 'D:/Msc.project/data_2017/training/instrument_dataset_1/instrument_dataset_1/ground_truth/validation_mask'
    
    pro_adjust_data(path_list, num_class, save_path)
    