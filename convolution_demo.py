import numpy as np
import cv2
from libs.util import MaskGenerator, ImageChunker
mask = MaskGenerator(128, 128, 3, rand_seed = 1222)._generate_mask()
mask=mask[0:63,0:63,0]

# import keras.activations as activations
# import tensorflow as tf
# f = np.array([[1, 2, 1],
#               [1, 0, 0],
#               [-1, 0, 1]])
# img = np.array([
#     [2, 3, 7, 4, 6, 2, 9],
#     [6, 6, 9, 8, 7, 4, 3],
#     [3, 4, 8, 3, 8, 9, 7],
#     [7, 8, 3, 6, 6, 3, 4],
#     [4, 2, 1, 8, 3, 4, 6],
#     [3, 2, 4, 1, 9, 8, 3],
#     [4, 5, 3, 9, 2, 1, 4]])
# img=np.random.randint(0,50,(63, 63))   这是生成随机数字的矩阵
input_img=cv2.imread(r"C:\Users\dell\Desktop\paper2\\figure\Fig3\\img.jpg")
# img0=np.array(input_img)
img=input_img[:,:,0]

masked_img=img*mask
kernel1=np.random.rand(7,7)
kernel2=np.random.rand(5,5)
kernel3=np.random.rand(3,3)
mask_kernel=np.ones([7,7])
# f=round(f,1)
strde=1

def cov2(img, f, strde):
    inw, inh = img.shape
    w, h = f.shape
    outw = int((inw - w) / strde + 1)
    outh = int((inh - h) / strde + 1)
    arr = np.zeros(shape=(outw, outh))
    for g in range(outh):
        for t in range(outw):
            s = 0
            for i in range(w):
                for j in range(h):
                    s += img[i + g * strde][j + t * strde] * f[i][j]
                    # s = img[i][j] * f[i][j]
            arr[g][t] = int(s)
    return arr


result1=cov2(masked_img,kernel1,strde)
result2=cov2(masked_img,kernel2,strde)
result3=cov2(masked_img,kernel3,strde)
mask_result=cov2(mask,mask_kernel,strde)
# result2=cov2(result1,f,1)
# result3=cov2(result2,f,1)
# print(img)
# print(kernel)
# print(result1)
# print(mask)
# np.savetxt(r'./RandomMatrix.txt',result1,fmt="%d", delimiter=',', header="行,"+"列",footer='By Accelerator')
np.savetxt(r'C:\Users\dell\Desktop\paper2\figure\Fig3\mask.txt',mask,fmt="%d", delimiter=',')
np.savetxt(r'C:\Users\dell\Desktop\paper2\figure\Fig3\mask_result.txt',mask_result,fmt="%d", delimiter=',')
np.savetxt(r'C:\Users\dell\Desktop\paper2\figure\Fig3\input.txt',img,fmt="%d", delimiter=',')
np.savetxt(r'C:\Users\dell\Desktop\paper2\figure\Fig3\masked_input.txt',masked_img,fmt="%d", delimiter=',')
np.savetxt(r'C:\Users\dell\Desktop\paper2\figure\Fig3\kernel1.txt',kernel1,fmt="%f", delimiter=',')
np.savetxt(r'C:\Users\dell\Desktop\paper2\figure\Fig3\kernel2.txt',kernel2,fmt="%f", delimiter=',')
np.savetxt(r'C:\Users\dell\Desktop\paper2\figure\Fig3\kernel3.txt',kernel3,fmt="%f", delimiter=',')
np.savetxt(r'C:\Users\dell\Desktop\paper2\figure\Fig3\result1.txt',result1,fmt="%d", delimiter=',')
np.savetxt(r'C:\Users\dell\Desktop\paper2\figure\Fig3\result2.txt',result2,fmt="%d", delimiter=',')
np.savetxt(r'C:\Users\dell\Desktop\paper2\figure\Fig3\result3.txt',result3,fmt="%d", delimiter=',')
# print(result2)
# print(result3)
# sess=tf.Session()
# img=np.expand_dims(img,0)
# # img_relu= activations.relu(img, alpha=0.0, max_value=None, threshold=4)
# print(img)
# print(sess.run(img_relu))