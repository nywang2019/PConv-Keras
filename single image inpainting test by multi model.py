# This file is to output and display single image inpainting result
# Created by wny 2019.10 Canada
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Change to root path
if os.path.basename(os.getcwd()) != 'PConv-Keras':
    os.chdir('..')

from libs.pconv_model import PConvUnet
#wny from libs.util import random_mask, ImageChunker
from libs.util import MaskGenerator, ImageChunker

# the following  lines are used in Jupyter, they are no more useful in Pycharm, so comment them.
# %load_ext autoreload
# %autoreload 2

# SETTINGS
#SAMPLE_IMAGE = 'data/sample_image11.jpg'
SAMPLE_IMAGE = 'data/000792.png'
BATCH_SIZE = 1

im = Image.open(SAMPLE_IMAGE)
im_width,im_height=im.size
# crop the original image to 512X 512
if im_height!=512 or im_width!=512:
    im=im.crop((im_width//2-512//2,im_height//2-512//2,im_width//2+512//2,im_height//2+512//2))
input_img=im
im = np.array(im) / 255

# The following first line is a mistake from the original file, replace it with the following second line.
#wny mask = random_mask(*crop)
mask = MaskGenerator(512,512)._generate_mask()

# This is to fuse mask to input image,and get a masked image
im[mask==0] = 1

# Create a model instance and import pre trained image_net weights provided by the author.
from libs.pconv_model import PConvUnet
model1 = PConvUnet(vgg_weights=None, inference_only=True)
model2 = PConvUnet(vgg_weights=None, inference_only=True)
model3 = PConvUnet(vgg_weights=None, inference_only=True)
model1.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\pconv_imagenet.26-1.07.h5", train_bn=False)
model2.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\weights.07-1.89.h5", train_bn=False)
model3.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\weights.10-1.74.h5", train_bn=False)
#output the predicted image
predicted_img1 = model1.predict([np.expand_dims(im, 0), np.expand_dims(mask, 0)])[0]
predicted_img2 = model2.predict([np.expand_dims(im, 0), np.expand_dims(mask, 0)])[0]
predicted_img3 = model3.predict([np.expand_dims(im, 0), np.expand_dims(mask, 0)])[0]
# This is to output both mask and masked image together.
fig, ax = plt.subplots(2,3)
ax[0][0].imshow(input_img)
ax[0][1].imshow(mask*255)
ax[0][2].imshow(im)
ax[1][0].imshow(predicted_img1)
ax[1][1].imshow(predicted_img2)
ax[1][2].imshow(predicted_img3)
plt.show()
