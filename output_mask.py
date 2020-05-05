
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from libs.util import MaskGenerator
# for i in range (0,100):
#     mask = MaskGenerator(512,512)._generate_mask()
#     #mask = MaskGenerator(256,256)._generate_mask()
#     # #mask = MaskGenerator(512, 512, 3, rand_seed = 666)._generate_mask()
#     cv2.imwrite('./temp_mask/mask_temp_'+str(i)+'.png', mask*255)

mask = MaskGenerator(512, 512, 3, rand_seed =22445)._generate_mask()
cv2.imwrite('./temp_mask/mask_temp_20200407.png', mask*255)