# this file is for evaluating PSNR,MSE and SSIM of result images that generated by different models.
# WNY 2019 Canada
import cv2
#import matplotlib.pyplot as plt
import os
#from copy import deepcopy
import numpy as np
from PIL import Image
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from libs.util import MaskGenerator
from libs.pconv_model import PConvUnet
from datetime import datetime

# original_img_folder=r'D:\PycharmProjects2\PConv-Keras\result_comparison\original_images'
original_img_folder=r'D:\PycharmProjects2\PConv-Keras\imgs\paper2_comparison'
# result_img_folder=r'D:\PycharmProjects2\PConv-Keras\imgs\paper2_comparison'
result_img_folder=r'C:\Users\dell\Desktop\paper2\figure'


def to_gray(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

model1 = PConvUnet(vgg_weights=None, inference_only=True)
# model2 = PConvUnet(vgg_weights=None, inference_only=True)
# model3 = PConvUnet(vgg_weights=None, inference_only=True)
# model4 = PConvUnet(vgg_weights=None, inference_only=True)
#model5 = PConvUnet(vgg_weights=None, inference_only=True)
#model6 = PConvUnet(vgg_weights=None, inference_only=True)
# model1.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t147\weights.01-0.65.h5", train_bn=False)
# model1.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t165\weights.46-1.23.h5", train_bn=False)
# model1.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t153\weights.52-0.31.h5", train_bn=False)
# model1.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t170\weights.26-1.19.h5", train_bn=False)
# model1.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t172\weights.15-1.18.h5", train_bn=False)
# model3.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t160\weights.12-1.53.h5", train_bn=False)
# model1.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t152\weights.31-1.20.h5", train_bn=False)
model1.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t34\weights.07-1.29.h5", train_bn=False)
#model6.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t36\weights.59-0.30.h5", train_bn=False)
# model1.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t16\weights.11-1.15.h5", train_bn=False)
models=[]
models.append(model1)
# models.append(model2)
# models.append(model3)
# models.append(model4)
#models.append(model5)
#models.append(model6)
mse=[0,0,0,0]
psnr=[0,0,0,0]
ssim=[0,0,0,0]
image_num=0

# use following 2 line to load a single mask to test all images:
mask=cv2.imread("./temp_mask.png")
mask=mask/255

# use following 2 line to generate a single mask to test all images:
# mask = MaskGenerator(512, 512, 3, rand_seed = 123)._generate_mask()
# cv2.imwrite('./output_mask_512_seed4210.png', mask*255)

start_time=datetime.now()
for filename in os.listdir(original_img_folder):
    image_num=image_num+1
    image = cv2.imread(os.path.join(original_img_folder, filename))
    input_img =image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = np.array(image) / 255
    #mask = MaskGenerator(512, 512)._generate_mask()
    #mask = MaskGenerator(512, 512, 3, rand_seed = 42)._generate_mask()
    image[mask == 0] = 1
    for j in range(0,len(models)):
        predicted_img=models[j].predict([np.expand_dims(image, 0), np.expand_dims(mask, 0)])[0]*255

        # if you want to save inpainted result, please use the following 3 lines, else use the 4th line below:
        result_name=result_img_folder+"\\"+filename.rstrip('.png') +"_"+str(j+1)+ ".png"
        cv2.imwrite(result_name, cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB))
        result_img=cv2.imread(result_name)

        # if you don't want to save inpainted result, please use the following line, else use the above 3 lines:
        # result_img=cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)

        current_mse=compare_mse(input_img,result_img)
        current_psnr=compare_psnr(input_img,result_img)
        current_ssim=compare_ssim(to_gray(input_img),to_gray(result_img))
        #print("MSE:{}".format(current_mse),"PSNR:{}".format(current_psnr),"SSIM:{}".format(current_ssim))
        print(f"Image Num:{image_num}({j+1})", f"MSE:{current_mse:4.4f}", f"PSNR:{current_psnr:3.4f}", f"SSIM:{current_ssim:3.4f}")
        mse[j]=mse[j]+current_mse
        psnr[j]=psnr[j]+current_psnr
        ssim[j]=ssim[j]+current_ssim

end_time=datetime.now()
total_time=(end_time-start_time).seconds

for k in range(0,len(models)):
     mse[k]=mse[k]/image_num
     psnr[k]=psnr[k]/image_num
     ssim[k]=ssim[k]/image_num
     #print("Average result of method:",k+1)
     #print("MSE:{}".format(mse[k]),"PSNR:{}".format(psnr[k]),"SSIM:{}".format(ssim[k]))
     print("Average result of method:",k+1,f"  MSE:{mse[k]:3.4f}", f"PSNR:{psnr[k]:3.4f}", f"SSIM:{ssim[k]:3.4f}")

print("tatal images:"+str(image_num))
print(f"Speed:{image_num*len(models)/total_time:1.2f} pic/s")
