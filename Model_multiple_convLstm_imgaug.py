"""
Created on Sat Mar 3 18:38:30 2019

Data processing for robotic Multiple image segmentation with ConvLSTM Unet model

@author: YONG Huawei
"""
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import backend as K
from PIL import Image
from imgaug import augmenters as iaa
import numpy as np
import tensorflow as tf
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320


def read_img(path, target_size, color_mode, multi_mask=False, num_class=1):
    try:
        img = Image.open(path).convert(color_mode)
    except Exception as e:
        print(e)
    else:  # this part can function only if the try is processing successfully
        img_arry = np.array(img)
        img_new = img_arry[h_start:h_start + height, w_start:w_start + width]
        img_arry = cv2.resize(img_new, target_size, interpolation=cv2.INTER_CUBIC)

        if multi_mask is True:
            # print(img_arry.shape)
            mask = np.zeros(shape=img_arry.shape + (num_class,))
            color_dict = [0, 50, 100, 150, 200]
            # defining the grey scale pixels for
            # Left_labels, Maryland_labels, Ot_labels, Right_labels seperatelly
            for i in range(num_class):
                mask[img_arry == color_dict[i], i] = 1  # background is one of the Five classes
            img_arry = mask

        else:
            img_arry = img_arry - np.mean(img_arry).astype('uint8')
            img_arry = np.reshape(img_arry, (img_arry.shape[0], img_arry.shape[1], 1))

        x = np.expand_dims(img_arry, axis=0)
        return x


def my_gen(time_seq, path, image_folder, mask_folder,
           batch_size, target_size=(256, 256)):
    #    img_list = glob.glob(path + '*.png')    # 获取path里面所有图片的路径
    image_path = path + '/' + image_folder
    mask_path = path + '/' + mask_folder
    img_list = os.listdir(image_path)
    msk_list = os.listdir(mask_path)
    steps = len(img_list) // time_seq  # get the integer to the floor
    #    print(steps)
    print("Found %s images." % len(img_list))
    print("Found %s masks." % len(msk_list))

    while True:
        time_array = []
        mask_array = []
        counter = 0
        for i in range(steps):
            counter += 1
            batch_list = img_list[i * time_seq: i * time_seq + time_seq]
            mask_list = msk_list[i * time_seq: i * time_seq + time_seq]
            x = [read_img(str(image_path + '/' + file), target_size, color_mode="L") for file in batch_list]
            y = [read_img(str(mask_path + '/' + file), target_size, color_mode="L", multi_mask=True, num_class=5) for file in
                 mask_list]
            # print(x[0][0].shape)
            # t = Image.fromarray(x[0][0])
            # t.show()
            batch_x = np.concatenate([array for array in x])
            batch_y = np.concatenate([arr for arr in y])
            # print("batch_x = %s" %(batch_x.shape)) (3, 256, 256, 1)
            # print("batch_y = %s" %(batch_y.shape))  # (3, 256, 256, 5)

            time_array.append(batch_x)
            mask_array.append(batch_y)

            augmenters_imgs = [iaa.Affine(rotate=(-10, 10)),
                               iaa.ElasticTransformation(sigma=0.2)]

            seq_imgs = iaa.Sequential(augmenters_imgs, random_order=False)

            if counter % batch_size == 0:
                seq_imgs_deterministic = seq_imgs.to_deterministic()  # call this everytime reset the seed
                for i in range(batch_size):
                    time_array[i] = np.expand_dims(np.array(seq_imgs_deterministic.augment_images(time_array[i])),
                                                   axis=0)
                    mask_array[i] = np.expand_dims(np.array(seq_imgs_deterministic.augment_images(mask_array[i])),
                                                   axis=0)
                    # print("batch_x = %s" %(batch_x.shape))  # (1, 3, 256, 256, 1)
                    # print("batch_y = %s" %(batch_y.shape))  # (1, 3, 256, 256, 5)
                train_x = np.concatenate([array for array in time_array])
                mask_y = np.concatenate([arr for arr in mask_array])

                time_array = []  # every batch_size cube then reset the list to zero
                mask_array = []

                # the mask shape (batch_size, sequence, 256, 256, 5)
                # the image shape (batch_size, seq, 256, 256, 5)

                yield train_x, mask_y


def validation_generator(time_seq, path, image_folder, mask_folder,
                         batch_size, target_size=(256,256)):
    image_path = path + '/' + image_folder
    mask_path = path + '/' + mask_folder
    img_list = os.listdir(image_path)
    msk_list = os.listdir(mask_path)
    steps = len(img_list) // time_seq  # get the integer to the floor
    #    print(steps)
    print("Found %s validation_images." % len(img_list))
    print("Found %s validation_masks." % len(msk_list))

    while True:
        time_array = []
        mask_array = []
        counter = 0
        for i in range(steps):
            counter += 1
            batch_list = img_list[i * time_seq: i * time_seq + time_seq]
            mask_list = msk_list[i * time_seq: i * time_seq + time_seq]
            x = [read_img(str(image_path + '/' + file), target_size, color_mode="L") for file in batch_list]
            y = [read_img(str(mask_path + '/' + file), target_size, color_mode="L", multi_mask=True, num_class=5) for file in
                 mask_list]

            batch_x = np.concatenate([array for array in x])
            batch_y = np.concatenate([arr for arr in y])
            # print(batch_x.shape)
            batch_x = np.expand_dims(batch_x, axis=0)  # expand the batch axis
            batch_y = np.expand_dims(batch_y, axis=0)
            # print(batch_x.shape)  # (1, 3, 256, 256)
            time_array.append(batch_x)
            mask_array.append(batch_y)
            if counter % batch_size == 0:
                train_x = np.concatenate([array for array in time_array])
                mask_y = np.concatenate([arr for arr in mask_array])
                print(len(mask_y))

                time_array = []  # per batch_size cube then reset the list to zero
                mask_array = []

                yield train_x, mask_y


def convLSTM_unet(pretrained_weights=None, input_size=(None, 256, 256, 1)):
    
    inputs = Input(input_size)
    conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(inputs)
    conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool1)
    conv2 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Conv2D(256, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal'))(pool2)
    conv3 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool3)
    conv4 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv4)
    drop4 = TimeDistributed(Dropout(0.5))(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(drop4)

    conv5 = TimeDistributed(Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool4)
    conv5 = TimeDistributed(Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv5)
    drop5 = TimeDistributed(Dropout(0.5))(conv5)

    up6 = ConvLSTM2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',return_sequences=True)(TimeDistributed(UpSampling2D(size=(2,2)))(drop5))

    merge6 = concatenate([drop4, up6], axis = 4)
    conv6 = ConvLSTM2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',return_sequences=True)(merge6)
    conv6 = ConvLSTM2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',return_sequences=True)(conv6)

    up7 = ConvLSTM2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',return_sequences=True)\
        (TimeDistributed(UpSampling2D(size = (2,2)))(conv6))
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7 = ConvLSTM2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',return_sequences=True)(merge7)
    conv7 = ConvLSTM2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',return_sequences=True)(conv7)

    up8 = ConvLSTM2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',return_sequences=True)\
        (TimeDistributed(UpSampling2D(size = (2,2)))(conv7))
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8 = ConvLSTM2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(merge8)
    conv8 = ConvLSTM2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(conv8)

    up9 = ConvLSTM2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',return_sequences=True)\
        (TimeDistributed(UpSampling2D(size = (2,2)))(conv8))
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9 = ConvLSTM2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(merge9)
    conv9 = ConvLSTM2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', return_sequences=True)(conv9)
    conv9 = TimeDistributed(Conv2D(5, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv9)
    conv10 = TimeDistributed(Conv2D(5, 1, activation = 'softmax'))(conv9)

    model = Model(input=inputs, output=conv10)

    # plot_model(model, to_file='seg_UNet3D.png', show_shapes=True)
    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def trainModel(train_path, image_folder, mask_folder,
               val_path, val_image_folder, val_mask_folder, time_seq, batch_size,
               mode_save_dir='unet.h5', num_image=225, epoch=60):

    # input image is sized into input_size = (None, 256, 256, 1) from the generator
    model = convLSTM_unet()
    # print("Training the model")
    model_checkpoint = ModelCheckpoint(mode_save_dir, monitor='loss', verbose=1, save_best_only=True)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss="categorical_crossentropy", metrics=['accuracy', mean_iou])

    model.fit_generator(my_gen(time_seq, train_path, image_folder, mask_folder, batch_size=batch_size),
                        steps_per_epoch=num_image/(batch_size*time_seq), epochs=epoch,
                        validation_data=validation_generator(time_seq,
                        val_path, val_image_folder, val_mask_folder, batch_size=batch_size),
                        validation_steps=num_image/(batch_size*time_seq), callbacks=[model_checkpoint])


if __name__ == '__main__':
    train_path = "/data/d0/ascstd/hwyong/data/train"
    image_folder = "left_frames"
    mask_folder = "pro_mask"
    val_path = "/data/d0/ascstd/hwyong/data/valid"
    val_image_folder = "val_train"
    val_mask_folder = "val_mask"
    model_path = '/data/ssd/public/MAX_YONG/EADseg/unet_multiple_convLSTM_6.h5'

    print("Main is functioning")

    trainModel(train_path, image_folder=image_folder, mask_folder=mask_folder,
               val_path=val_path, val_image_folder=val_image_folder,
               val_mask_folder=val_mask_folder, time_seq=2,
               mode_save_dir=model_path, batch_size=2, num_image=225, epoch=200)
