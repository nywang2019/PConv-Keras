from PIL import Image, ImageDraw
import numpy as np
import cv2
# img1=Image.open('./imgs/016.png')
# img2=Image.open('./imgs/generated_input_image.png')
# print(img1.size)
# print(img2.size)
# img1=cv2.imread('./imgs/generated_mask_image.png')
# img2=cv2.imread('./imgs/generated_input_image.png')
# print(img1.shape)
# print(img2.shape)
# img1 = np.asarray(img1,dtype='int');
# cv2.imwrite('./imgs/016_2.png', np.asarray(img1).transpose(2, 1, 0))
#
# im = Image.open("./imgs/016.png");
#
# image = np.asarray(im,dtype='int');
#
# out = image.transpose(1,0,2);
#
# p = Image.fromarray(np.uint8(out));
#
# p.save("./imgs/016-2.png");
mask=cv2.imread("mask4.jpg")
print(mask.shape)
mask = np.minimum(mask, 1.0)
#mask=1-mask/255
#m= np.asarray(mask[0,:,:],dtype='int');
for i in range(1,511):
    b=mask[i]
    print(b)
