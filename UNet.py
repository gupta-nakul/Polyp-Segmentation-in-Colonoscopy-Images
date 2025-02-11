#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision.io import read_image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# In[3]:


import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import CustomObjectScope


# In[4]:


from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model


# In[5]:


from tensorflow.keras.utils import Sequence


# In[6]:


def create_images_list(path):
    full_path = []
    images = sorted(os.listdir(path))

    for i in images:
        full_path.append(os.path.join(path, i))

    return full_path


# In[7]:


def parse_image(img_path, image_size):
    image_rgb = cv2.imread(img_path, 1)
    h, w, _ = image_rgb.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        image_rgb = cv2.resize(image_rgb, (image_size, image_size))
    image_rgb = image_rgb/255.0
    return image_rgb

def parse_mask(mask_path, image_size):
    mask = cv2.imread(mask_path, -1)
    h, w = mask.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        mask = cv2.resize(mask, (image_size, image_size))
    mask = np.expand_dims(mask, -1)
    mask = mask/255.0

    return mask


# In[8]:


class DataGen(Sequence):
    def __init__(self, image_size, images_path, masks_path, batch_size=8):
        self.image_size = image_size
        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.on_epoch_end()

    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.images_path):
            self.batch_size = len(self.images_path) - index*self.batch_size

        images_path = self.images_path[index*self.batch_size : (index+1)*self.batch_size]
        masks_path = self.masks_path[index*self.batch_size : (index+1)*self.batch_size]

        images_batch = []
        masks_batch = []

        for i in range(len(images_path)):
            ## Read image and mask
            image = parse_image(images_path[i], self.image_size)
            mask = parse_mask(masks_path[i], self.image_size)

            images_batch.append(image)
            masks_batch.append(mask)

        return np.array(images_batch), np.array(masks_batch)

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.images_path)/float(self.batch_size)))


# In[9]:


def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


# In[10]:


def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same")(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


# In[11]:


def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


# In[12]:


def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding="same")(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding="same")(y)
    return y


# In[13]:


def attetion_block(g, x):
    """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
    """

    filters = x.shape[-1]

    g_conv = BatchNormalization()(g)
    g_conv = Activation("relu")(g_conv)
    g_conv = Conv2D(filters, (3, 3), padding="same")(g_conv)

    g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

    x_conv = BatchNormalization()(x)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(filters, (3, 3), padding="same")(x_conv)

    gc_sum = Add()([g_pool, x_conv])

    gc_conv = BatchNormalization()(gc_sum)
    gc_conv = Activation("relu")(gc_conv)
    gc_conv = Conv2D(filters, (3, 3), padding="same")(gc_conv)

    gc_mul = Multiply()([gc_conv, x])
    return gc_mul


# In[14]:


class ResUnetPlusPlus:
    def __init__(self, input_size=256):
        self.input_size = input_size

    def build_model(self):
        n_filters = [16, 32, 64, 128, 256]
        inputs = Input((self.input_size, self.input_size, 3))

        c0 = inputs
        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)

        ## Bridge
        b1 = aspp_block(c4, n_filters[4])

        ## Decoder
        d1 = attetion_block(c3, b1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attetion_block(c2, d1)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attetion_block(c1, d2)
        d3 = UpSampling2D((2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])

        ## output
        outputs = aspp_block(d3, n_filters[0])
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs)
        return model


# In[15]:


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

# In[ ]:


if __name__ == '__main__':
    train_images = create_images_list('./datasets/kvasir_segmentation_dataset/kvasir_segmentation_dataset/train/images')
    train_masks = create_images_list('./datasets/kvasir_segmentation_dataset/kvasir_segmentation_dataset/train/masks')
    valid_images = create_images_list('./datasets/kvasir_segmentation_dataset/kvasir_segmentation_dataset/valid/images')
    valid_masks = create_images_list('./datasets/kvasir_segmentation_dataset/kvasir_segmentation_dataset/valid/masks')
    print(len(train_images), len(train_masks))

    train_data = pd.DataFrame({'image': train_images, 'mask': train_masks})
    train_data = shuffle(train_data).reset_index().drop(columns=['index'])
    
    val_data = pd.DataFrame({'image': valid_images, 'mask': valid_masks})
    val_data = shuffle(val_data).reset_index().drop(columns=['index'])
    
    X_train = train_data['image']
    y_train = train_data['mask']
    
    X_validation = val_data['image']
    y_validation = val_data['mask']
    
    ## Parameters
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 1
    
    ## Generator
    train_gen = DataGen(image_size, list(X_train), list(y_train), batch_size=batch_size)
    valid_gen = DataGen(image_size, list(X_validation), list(y_validation), batch_size=batch_size)

    ## ResUnet++
    arch = ResUnetPlusPlus(input_size=image_size)

    model = arch.build_model()

    optimizer = Nadam(lr)
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
    model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)

    # csv_logger = CSVLogger(f"{file_path}unet_{batch_size}.csv", append=False)
    # checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    # callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]
    callbacks = [reduce_lr, early_stopping]

    train_steps = len(X_train) // batch_size
    valid_steps = len(X_validation) // batch_size

    #model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
          #epochs=epochs, callbacks=callbacks)

    #model.save('resunetplusplus.h5')

    with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
        model = load_model('resunetplusplus.h5')
    
    test_images = create_images_list('./datasets/kvasir_segmentation_dataset/kvasir_segmentation_dataset/test/images')
    test_masks = create_images_list('./datasets/kvasir_segmentation_dataset/kvasir_segmentation_dataset/test/masks')

    test_data = pd.DataFrame({'image': test_images, 'mask': test_masks})
    test_data = shuffle(test_data).reset_index().drop(columns=['index'])

    X_test = test_data['image']
    y_test = test_data['mask']

    image_size = 256
    batch_size = 1

    # Test
    test_steps = len(X_test)//batch_size
    test_gen = DataGen(image_size, list(X_test), list(y_test), batch_size=batch_size)

    print("Test Result: ")
    model.evaluate(test_gen, steps=test_steps, verbose=1)
    save_path = "./output"

    ### Generating the result
    for i, path in tqdm(enumerate(list(X_test)), total=len(list(X_test))):
        image = parse_image(list(X_test)[i], image_size)
        mask = parse_mask(list(y_test)[i], image_size)

        predict_mask = model.predict(np.expand_dims(image, axis=0))[0]
        predict_mask = (predict_mask > 0.5) * 255.0

        sep_line = np.ones((image_size, 10, 3)) * 255

        mask = mask_to_3d(mask)
        predict_mask = mask_to_3d(predict_mask)

        all_images = [image * 255, sep_line, mask * 255, sep_line, predict_mask]
        img_name = path.split("/")[-1][:-4]
        cv2.imwrite(f"{save_path}/{img_name}.jpg", np.concatenate(all_images, axis=1))

    print("Test image generation complete")



# In[ ]:


# ## Create result folder
# try:
#     os.mkdir(save_path)
# except:
#     pass

# ## Model
# # with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
# #     model = load_model(model_path)





# In[ ]:




