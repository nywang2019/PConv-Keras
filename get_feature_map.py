

import numpy as np
# from processor import process_image
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
import cv2


def main():
    model = load_model('./data/logs/Thanka_phase1/p1t152/weights.01-1.21.h5')  # replaced by your model name
    # Get all our test images.
    image = 'ori6.png'
    images = cv2.imread(r"D:\PycharmProjects2\PConv-Keras\imgs\ori6.png")
    cv2.imshow("Image", images)
    cv2.waitKey(0)
    # Turn the image into an array.
    # image_arr = process_image(image, (299, 299, 3))  # 根据载入的训练好的模型的配置，将图像统一尺寸
    image_arr = np.expand_dims(images, axis=0)

    # 设置可视化的层
    layer_1 = K.function([model.layers[0].input], [model.layers[1].output])
    f1 = layer_1([image_arr])[0]
    for _ in range(32):
        show_img = f1[:, :, :, _]
        show_img.shape = [256, 256]
        plt.subplot(4, 8, _ + 1)
        plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()
    # conv layer: 299
    layer_1 = K.function([model.layers[0].input], [model.layers[7].output])
    f1 = layer_1([image_arr])[0]
    for _ in range(32):
        show_img = f1[:, :, :, _]
        show_img.shape = [256, 256]
        plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()
    print('This is the end !')


if __name__ == '__main__':
    main()
