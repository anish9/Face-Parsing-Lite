# Face-Parsing-Lite - Tensorflow

![results](https://github.com/anish9/Face-Parsing-Lite/blob/main/assets/result_col.png)


### Dataset
The masks of CelebAMask-HQ were manually-annotated with the size of 512 x 512 and 19 classes including all facial components and accessories such as skin, nose, eyes, eyebrows, ears, mouth, lip, hair, hat, eyeglass, earring, necklace, neck, and cloth.

### Modified Light-Weight Architecture.
* ```Invert residual blocks``` design was used for Squeeze and Exapansion.
* ```DepthWise Convolution``` was used for the balance of parameter efficiency and accuracy.
* ```gelu``` was used in upscale pathway for smooth graident.
*  Trained with a ```Polynomial Decay Schedule```.
### Acknowledge
https://github.com/zllrunning/face-parsing.PyTorch
