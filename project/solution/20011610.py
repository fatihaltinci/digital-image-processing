#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz')
get_ipython().system('wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz')


# In[2]:


get_ipython().system('tar --gunzip --extract --verbose --file=images.tar.gz')
get_ipython().system('tar --gunzip --extract --verbose --file=annotations.tar.gz')


# In[3]:


import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

input_dir = "images/"
target_dir = "annotations/trimaps/"
img_size = (128, 128)
num_classes = 3
batch_size = 32
random_state = 7
train_size, val_size = 0.7, 0.3

imageNames = [os.path.basename(file) for file in glob.glob(os.path.join(input_dir, '*.jpg'))]

print(f"\nTotal number of image files: {len(imageNames)}")

labels = [' '.join(name.split('_')[:-1]) for name in imageNames ]

print(f"\nTotal number of unique labels: {len(np.unique(labels))}")


# In[4]:


labelEncDict = {name : ind for ind, name in enumerate(np.unique(labels))}
for k, v in labelEncDict.items():
    print(f"{k:32} : {v}")


# In[5]:


labelDecDict = {ind: name for name, ind in labelEncDict.items()}
for k, v in labelDecDict.items():
    print(f"{k:3} : {v}")


# In[6]:


for i in labelEncDict.keys():
    print(f"{i:32} : {labels.count(i)}")


# In[7]:


from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
from tensorflow import keras
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Add, Dense, Conv2DTranspose, MaxPool2D
from keras.layers.core import Flatten, Reshape
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps
import random


# In[8]:


from tensorflow.keras import layers

def LinkNet(img_height, img_width, nclasses=None):

  encoder_output_maps = {
        'block_1': 64,
        'block_2': 128,
        'block_3': 256,
        'block_4': 512
    }
  decoder_output_maps = {
      'block_1': 64,
      'block_2': 64,
      'block_3': 128,
      'block_4': 256
  }
  encoder_input_maps = {
      'block_1': 64,
      'block_2': 64,
      'block_3': 128,
      'block_4': 256
  }
  decoder_input_maps = {
      'block_1': 64,
      'block_2': 128,
      'block_3': 256,
      'block_4': 512
  }

  def encoder_block(input_tensor, block=None):
    block_name = f'block_{block}'
    nfilters = encoder_output_maps[block_name]
    input_tensor_projection = layers.Conv2D(nfilters, kernel_size=(1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal', name=f'{block_name}_conv2d_1x1')(input_tensor)
    input_tensor_projection = layers.BatchNormalization(name=f'{block_name}_bn_1x1')(input_tensor_projection)
    input_tensor_projection = layers.Activation('relu', name=f'{block_name}_relu_1x1')(input_tensor_projection)
    x = layers.Conv2D(nfilters, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name=f'encoder_{block_name}_conv2d_1')(input_tensor)
    x = layers.BatchNormalization(name=f'encoder_{block_name}_bn_1')(x)
    x = layers.Activation('relu', name=f'encoder_{block_name}_relu_1')(x)
    x = layers.Conv2D(nfilters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', name=f'encoder_{block_name}_conv2d_2')(x)
    x = layers.BatchNormalization(name=f'encoder_{block_name}_bn_2')(x)
    x = layers.Add(name=f'encoder_{block_name}_add_1')([x, input_tensor_projection])
    x = layers.Activation('relu', name=f'encoder_{block_name}_relu_2')(x)
    x_res = x
    x = layers.Conv2D(nfilters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', name=f'encoder_{block_name}_conv2d_3')(x_res)
    x = layers.BatchNormalization(name=f'encoder_{block_name}_bn_3')(x)
    x = layers.Activation('relu', name=f'encoder_{block_name}_relu_3')(x)
    x = layers.Conv2D(nfilters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', name=f'encoder_{block_name}_conv2d_4')(x)
    x = layers.BatchNormalization(name=f'encoder_{block_name}_bn_4')(x)
    x = layers.Add(name=f'encoder_{block_name}_add_2')([x, x_res])
    x = layers.Activation('relu', name=f'encoder_{block_name}_relu_4')(x)
    return x


  def decoder_block(input_tensor, block=None):
    block_name = f'block_{block}'
    nfilters_b = decoder_output_maps[block_name]
    nfilters_a = decoder_input_maps[block_name] // 4
    x = layers.Conv2D(nfilters_a, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', name=f'decoder_{block_name}_conv2d_1x1')(input_tensor)
    x = layers.BatchNormalization(name=f'decoder_{block_name}_bn_1')(x)
    x = layers.Activation('relu', name=f'decoder_{block_name}_relu_1')(x)
    x = layers.Conv2DTranspose(nfilters_a, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name=f'decoder_{block_name}_conv2d_transpose_1')(x)
    x = layers.BatchNormalization(name=f'decoder_{block_name}_bn_2')(x)
    x = layers.Activation('relu', name=f'decoder_{block_name}_relu_2')(x)
    x = layers.Conv2D(nfilters_b, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', name=f'decoder_{block_name}_conv2d_1')(x)
    x = layers.BatchNormalization(name=f'decoder_{block_name}_bn_3')(x)
    x = layers.Activation('relu', name=f'decoder_{block_name}_relu_3')(x)
    x = layers.Conv2D(nfilters_b, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', name=f'decoder_{block_name}_conv2d_2')(x)
    x = layers.BatchNormalization(name=f'decoder_{block_name}_bn_4')(x)
    return x

  input_tensor = Input(shape=(img_height, img_width, 3))
  x = input_tensor
  for i in range(1, 5):
      x = encoder_block(x, i)
  for i in range(1, 5):
      x = decoder_block(x, i)
  output = Conv2D(filters=nclasses, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='output_conv2d_1x1')(x)
  output = BatchNormalization(name='output_bn_1x1')(output)
  output = Activation('softmax', name='output_softmax')(output)
  model = Model(inputs=input_tensor, outputs=output)
  return model


# In[9]:


from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y


# In[10]:


import random

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

val_samples = int(len(input_img_paths)*0.3)
train_samples = len(input_img_paths) - val_samples
random.Random(42).shuffle(input_img_paths)
random.Random(42).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:train_samples]
train_target_img_paths = target_img_paths[:train_samples]
val_input_img_paths = input_img_paths[train_samples:]
val_target_img_paths = target_img_paths[train_samples:]


# In[11]:


train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)


# In[12]:


model = LinkNet(128, 128, num_classes)
model.summary()


# In[ ]:


# def dice_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#   union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
#   dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
#   return dice

# Bu fonksiyon da kullanılabilir ama one hot encoding yapılmadığı için sürekli sabit değer veriyor arkaplan/ve kategorize edilmemiş label'ı ayıramıyor


# In[ ]:


# def dice_coef(y_true, y_pred, smooth=1):
#     # flatten
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     # one-hot encoding y with 3 labels : 0=background, 1=label1, 2=label2
#     y_true_f = K.one_hot(K.cast(y_true_f, np.uint8), 3)
#     y_pred_f = K.one_hot(K.cast(y_pred_f, np.uint8), 3)
#     # calculate intersection and union exluding background using y[:,1:]
#     intersection = K.sum(y_true_f[:,1:]* y_pred_f[:,1:], axis=[-1])
#     union = K.sum(y_true_f[:,1:], axis=[-1]) + K.sum(y_pred_f[:,1:], axis=[-1])
#     # apply dice formula
#     dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
#     return dice
# def dice_loss(y_true, y_pred):
#   return 1-dice_coef

# Bu fonksiyon aktivasyon fonksiyonu sigmoid olmadığı için hata veriyor ya da Flatten layer eklenmesi gerek


# In[ ]:


# def dice_coef(y_true, y_pred):
#   class_wise_dice_score = []

#   smoothening_factor = 0.00001

#   for i in range(3):
#     intersection = K.sum((y_pred == i, np.uint8) * (y_true == i, np.uint8))
#     y_true_area = K.sum((y_true == i, np.uint8))
#     y_pred_area = K.sum((y_pred == i,np.uint8))
#     combined_area = y_true_area + y_pred_area
    
#     dice_score =  2 * ((intersection + smoothening_factor) / (combined_area + smoothening_factor))
#     class_wise_dice_score.append(dice_score)

#     return class_wise_dice_score


# In[13]:


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


# In[14]:


model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics = ["accuracy", "sparse_categorical_accuracy", dice_coef])

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]
epochs = 10
history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)


# In[15]:


save_directory = 'ciktilar'
if not os.path.exists(save_directory):
    os.mkdir(save_directory)

model.save(os.path.join(save_directory,'goruntuislemeproje.h5'))


# In[16]:


acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

epochs_range = history.epoch

plt.figure(figsize = (16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.show()


# In[18]:


dice = history.history['dice_coef']
val_dice = history.history['val_dice_coef']

epochs_range = range(1, epochs+1)
plt.figure(figsize = (16, 8))
plt.plot(epochs_range, dice, label='Training Dice Coef')
plt.plot(epochs_range, val_dice, label='Validation Dice Coef')
plt.legend(loc='upper right')
plt.title('Training and Validation Dice Coef')
plt.xlabel('Epochs')
plt.ylabel('Dice Coef')
plt.show()


# In[22]:


val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)
i = 5
display(Image(filename=val_input_img_paths[i]))
img = ImageOps.autocontrast(load_img(val_target_img_paths[i]))
display(img)
display_mask(i)


# In[26]:


validation_data = val_gen

validation_steps = len(validation_data)

score = model.evaluate(validation_data, steps = validation_steps, verbose=0)

print("Validation Loss: ", score[0])
print("Validation Accuracy: ", score[3])
print("Validation Dice Coefficient: ", score[2])

