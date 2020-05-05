import cv2
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
psnr1,psnr2,psnr3,psnr4=0,0,0,0
ssim1,ssim2,ssim3,ssim4=0,0,0,0
def to_gray(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
for i in range(1,8):
    img0 = cv2.imread("./together/" + str(i) + ".png")
    img1 = cv2.imread("./together/" + str(i) + "_mine.png")
    img2 = cv2.imread("./together/" + str(i) + "_pconv.png")
    img3 = cv2.imread("./together/" + str(i) + "_gmcnn.png")
    img4 = cv2.imread("./together/" + str(i) + "_deepfill.png")
    current_psnr1 = compare_psnr(img0, img1)
    current_psnr2 = compare_psnr(img0, img2)
    current_psnr3 = compare_psnr(img0, img3)
    current_psnr4 = compare_psnr(img0, img4)
    current_ssim1 = compare_ssim(to_gray(img0), to_gray(img1))
    current_ssim2 = compare_ssim(to_gray(img0), to_gray(img2))
    current_ssim3 = compare_ssim(to_gray(img0), to_gray(img3))
    current_ssim4 = compare_ssim(to_gray(img0), to_gray(img4))
    psnr1=  psnr1+  current_psnr1
    psnr2 = psnr2 + current_psnr2
    psnr3 = psnr3 + current_psnr3
    psnr4 = psnr4 + current_psnr4
    ssim1 = ssim1+  current_ssim1
    ssim2 = ssim2 + current_ssim2
    ssim3 = ssim3 + current_ssim3
    ssim4 = ssim4 + current_ssim4
print(psnr1/7,psnr2/7,psnr3/7,psnr4/7)
print(ssim1/7,ssim2/7,ssim3/7,ssim4/7)

