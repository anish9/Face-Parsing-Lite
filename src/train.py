import os
import numpy as np
from glob import glob
import tensorflow as tf
import cv2
import distinctipy

from modules import *
from tensorflow.keras.callbacks import ModelCheckpoint


import matplotlib.pyplot as plt
#%matplotlib inline

os.environ["KERAS_BACKEND"] = "tensorflow"

#change paths 
dataset_train_image = "parse_dataset/dataset_final/train/image/" 
dataset_train_mask  = "parse_dataset/dataset_final/train/mask/"
dataset_val_image   = "parse_dataset/dataset_final/val/image/"
dataset_val_mask    = "parse_dataset/dataset_final/val/mask/"

train_images = [dataset_train_image+file for file in os.listdir(dataset_train_image) if file.endswith(".jpg") | file.endswith(".jpeg") | file.endswith(".png")]
train_masks  = [f.replace(dataset_train_image,dataset_train_mask).split(".j")[0]+".png" for f in train_images]
val_images   = [dataset_val_image+file for file in os.listdir(dataset_val_image) if file.endswith(".jpg") | file.endswith(".jpeg") | file.endswith(".png")]
val_masks    = [f.replace(dataset_val_image,dataset_val_mask).split(".j")[0]+".png" for f in val_images]


train_set = tf.data.Dataset.from_tensor_slices((train_images,train_masks))
val_set   = tf.data.Dataset.from_tensor_slices((train_images,train_masks))

def random_cropper(img_array,mask_array):
    MAX_SCALE = 800
    crop_min_range,crop_max_range =720,790
    img_array  = tf.image.resize_with_pad(tf.cast(img_array,tf.float32),MAX_SCALE,MAX_SCALE)
    mask_array = tf.image.resize_with_pad(tf.cast(mask_array,tf.float32),MAX_SCALE,MAX_SCALE,method="nearest")
    concat_img_mask = tf.concat((img_array,mask_array),axis=-1)
    crop_size = np.random.randint(crop_min_range,crop_max_range,1)[0]
    random_crop = tf.image.random_crop(concat_img_mask,size=(crop_size,crop_size,4))
    img_array,mask_array = random_crop[:,:,:3],random_crop[:,:,-1]
    return img_array,mask_array[:,:,tf.newaxis]


def prepare_train(image,mask):
    img  = tf.io.decode_image(tf.io.read_file(image),channels=3)
    mask = tf.io.decode_png(tf.io.read_file(mask),channels=1)
    img = tf.image.random_brightness(img,0.2)
    crop_img,crop_mask = random_cropper(img,mask)
    img,mask = tf.image.resize_with_pad(crop_img,512,512),tf.image.resize_with_pad(crop_mask,512,512,method="nearest")
    return img/255.,mask

def prepare_val(image,mask):
    img  = tf.io.decode_image(tf.io.read_file(image),channels=3)
    mask = tf.io.decode_png(tf.io.read_file(mask),channels=1)
    img,mask = tf.image.resize_with_pad(img,512,512),tf.image.resize_with_pad(mask,512,512,method="nearest")
    return img/255.,mask


model_base = parser_network(nclasses=19)

batch_size = 12
epochs = 18
starter_learning_rate = 1e-4
end_learning_rate = 2e-5
decay_steps = (len(train_images)//2)*epochs
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.5)
artifact_name = "bestmodel"

traindata = train_set.map(prepare_train,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(12,drop_remainder=True)
valdata   = val_set.map(prepare_val,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(12,drop_remainder=True)


model_base.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate_fn),
                   loss="sparse_categorical_crossentropy",metrics=["accuracy"])
ckpt = ModelCheckpoint(f"{artifact_name}.weights.h5",save_best_only=True,
                       save_weights_only="True",monitor="val_accuracy")
model_base.fit(traindata,batch_size=batch_size,epochs=epochs,validation_data=valdata,
               validation_steps=len(val_images)//batch_size,callbacks=[ckpt])
