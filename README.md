# Image-generator
## Keras image data generator with new additional transform functions for palimpsests OCR.
### (Functions can only be used for image where channel dimension is last, data_format='channels_last', and Keras version 2.0)

Functions include elastic transform and MNIST mask overlay.
To use this module instead of Keras "image.py" you need to download this file into your working folder and import ImageDataGenerator class from new image.py module:
-from image import ImageDataGenerator


To use elastic transform and MNIST mask overlay functions for image augmentation you need to indicate `elastic_RGB` and `MNIST_aug` parameters:
train_datagen = ImageDataGenerator(MNIST_aug=['./path/margins', [2,2],[50,150,255]],elastic_RGB=[4,1.6])


MNIST_aug parameters:
  MNIST_overlay(margins_dirpath,ls_pieces,mask_transparency,data_format='channels_last')
    '''
    Overlay image with MNIST mask split in pieces and filled with pixels from page margin patches
    data_format - should be 'channels_last'
    margins - dir path to folder with patches from page margins
    ls_pieces - list of numbers that overlayed mask would be divided, length of the list defines number of repeats
    mask_transparency - list of transparency level values that would be randomly set to overlaying layers
    '''
    
    
  elastic_transform_RGB(alpha, sigma, data_format='channels_last', random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       Works only for "channel_last" data format, 
       For original characters size 60-100 pixels use parameters [alpha,sigma]=[4,1.9]  
       Elastic transform RGB is modernized version of elastic transform taken from [link](https://gist.github.com/erniejunior/601cdf56d2b424757de5)
    """
