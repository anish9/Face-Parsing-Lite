import tensorflow as tf
from tensorflow.keras import layers
activation_ = "relu"


class upscale_block(layers.Layer):
    def __init__(self,transpose_filter,convfilter):
        super().__init__()
        self.convt = layers.Conv2DTranspose(transpose_filter,3,strides=2,padding="same")
        self.conv  = layers.Conv2D(convfilter,3,padding="same")
        self.bn1   = layers.BatchNormalization()
        self.act1  = layers.Activation("gelu")
        self.act2  = layers.Activation(activation_)

    def call(self,x1,x2):
        x = self.act1(self.convt(x1))
        x = layers.concatenate((x,x2))
        x = self.act2(self.bn1(self.conv(x)))
        return x
        

     
class inv_res_block(layers.Layer):
    def __init__(self,filters,stride):
        super().__init__()
        self.pw1 =  layers.Conv2D(filters=filters,kernel_size=(1,1),strides=stride,padding="same")
        self.dw1 =  layers.DepthwiseConv2D(kernel_size=(3,3),padding="same")
        self.pw2 =  layers.Conv2D(filters,kernel_size=(1,1),padding="same")
        self.bn1 =  layers.BatchNormalization()
        self.bn2 =  layers.BatchNormalization()
        self.bn3 =  layers.BatchNormalization()
        self.act1=  layers.Activation(activation_)
        self.act2=  layers.Activation(activation_)
        self.act3=  layers.Activation(activation_)

    def call(self,x):
        x0 = self.act1(self.bn1(self.pw1(x)))
        x1 = self.act2(self.bn2(self.dw1(x0)))
        x2 = self.act3(self.bn3(self.pw2(x1)))
        return tf.add(x0,x2)


def parser_network(nclasses):
    
    input_node = layers.Input(shape=(None,None,3))
    
    x1 = inv_res_block(filters=16,stride=1)(input_node)
    x2 = inv_res_block(filters=32,stride=2)(x1)
    x3 = inv_res_block(filters=64,stride=2)(x2)
    x4 = inv_res_block(filters=96,stride=2)(x3)
    
    x5 = inv_res_block(filters=96,stride=1)(x4)
    x6 = inv_res_block(filters=128,stride=2)(x5)
    x7 = inv_res_block(filters=128,stride=2)(x6)
    x8 = inv_res_block(filters=192,stride=2)(x7)
    
    b = layers.BatchNormalization()(layers.DepthwiseConv2D((3,3),padding="same")(x8))
    
    u1 = upscale_block(transpose_filter=128,convfilter=128)(b,x7)
    u2 = upscale_block(transpose_filter=96,convfilter=64)(u1,x6)
    u3 = upscale_block(transpose_filter=64,convfilter=64)(u2,x5)
    u4 = upscale_block(transpose_filter=32,convfilter=64)(u3,x3)
    u5 = upscale_block(transpose_filter=32,convfilter=32)(u4,x2)
    u6 = upscale_block(transpose_filter=32,convfilter=32)(u5,x1)

    out = layers.Conv2D(nclasses,1,padding="same")(u6)
    if nclasses>1:
        out = layers.Activation("softmax")(out)
    else:
        out = layers.Activation("sigmoid")(out)
    
    model_ = tf.keras.models.Model(input_node,out,name="parser")
    return model_
    