import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from src.modules import *
import matplotlib.pyplot as plt
os.environ["KERAS_BACKEND"] = "tensorflow"


model_base = parser_network(nclasses=19)
model_base.load_weights("faceparserV1.weights.h5")



def predict_visualize(image_path: str):
    """This function produces pixelwise segmentation visualzation.
    Args:
      image_path
    Returns:
      prediction of tensor(Segmentation Mask) of size (1,512,512,nlabels).
      Example here: (1,512,512,18)
      """
    img  = tf.io.decode_image(tf.io.read_file(image_path),channels=3)
    img = tf.image.resize_with_pad(img,512,512)
    copy_rgb = img
    img = img[tf.newaxis,:,:,:]/255.
    pred = model_base(img)
    mask_tensor = tf.argmax(pred[0],axis=-1)
    mask_tensor = tf.repeat(mask_tensor[:,:,tf.newaxis],repeats=3,axis=-1).numpy()
    colors = np.array([[197,   4, 254],[  0, 255,   0],[250, 119,   3],[ 37, 145, 129],
                       [  0,   10, 225],[129, 125,  93],[122,  85, 123],[  0, 127,   0],
                       [255, 127, 255],[192,   2,  11],[  0,   0, 127],[  0, 255, 127],
                       [255, 255,   0],[127, 245, 236],[249,  58, 135],[  5, 141, 115],
                       [108,  87, 254],[127, 127,   0],[167, 158, 191],[252, 175, 103]], dtype=np.uint8)
    
    dicts = dict(zip(range(0,19),colors))
    copy = mask_tensor.copy()*0
    for j,ele in enumerate(np.unique(mask_tensor)):
        copy = np.where(mask_tensor==ele,copy+dicts[ele],copy)
    
    segment_mask = copy
    segment_mask = segment_mask.astype(np.uint8)
    visualize = cv2.addWeighted(copy_rgb.numpy().astype(np.uint8),0.4,segment_mask,0.6,1.0)
    cv2.imwrite("visualize.png",visualize)
    return pred


if __name__ == "__main__":
    pred = predict_visualize("sample.jpg")