# This file is to interactively inpaint image and display image inpainting result in realtime.
# Created by wny 2019.10 Canada
from tkinter import *
from PIL import Image, ImageTk, ImageDraw
import tkinter.filedialog as tkFileDialog
import numpy as np
import cv2
import os
import subprocess
import argparse
import tensorflow as tf
from options.test_options import TestOptions


##################################################################
# wny from net.network import GMCNNModel
from libs.pconv_model import PConvUnet
# wny model = PConvUnet(vgg_weights=None, inference_only=True)
##################################################################


# os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
#     "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# model_savepath = {'places2ful': 'model_places2ful512/', 'places2': 'model_places2ful/', 'celebAHQ': 'model_celebAHQ/',
#                   'paris_streetview': 'model_paris/', 'celebA': 'model_celebA/'}

class Paint(object):
    MARKER_COLOR = 'white'

    def __init__(self, config):
        self.config = config

        self.root = Tk()
        self.root.title("Image Inpainting (V1.0)")
        # self.LabelArea=Label(text="中国", bg="green", font=("Arial", 12), width=10, height=2)
        # self.LabelArea.grid(row=1, column=0, rowspan=1)

        self.c = Canvas(self.root, bg='white', width=config.img_shapes[1] + 8, height=config.img_shapes[0])
        self.c.grid(row=0, column=0, rowspan=8)

        self.out = Canvas(self.root, bg='white', width=config.img_shapes[1] + 8, height=config.img_shapes[0])
        self.out.grid(row=0, column=1, rowspan=8)

        self.out2 = Canvas(self.root, bg='white', width=config.img_shapes[1] + 8, height=config.img_shapes[0])
        self.out2.grid(row=0, column=2, rowspan=8)

        self.load_button = Button(self.root, text='load', command=self.load, width=12, height=3)
        self.load_button.grid(row=0, column=3)

        self.rect_button = Button(self.root, text='rectangle', command=self.use_rect, width=12, height=3)
        self.rect_button.grid(row=1, column=3)

        self.poly_button = Button(self.root, text='stroke', command=self.use_poly, width=12, height=3)
        self.poly_button.grid(row=2, column=3)

        self.fill_button = Button(self.root, text='fill', command=self.fill, width=12, height=3)
        self.fill_button.grid(row=3, column=3)

        self.clear_button = Button(self.root, text='clear', command=self.clear, width=12, height=3)
        self.clear_button.grid(row=4, column=3)

        self.revoke_button = Button(self.root, text='    ', command=self.revoke, width=12, height=3)
        self.revoke_button.grid(row=5, column=3)

        self.save_button = Button(self.root, text="save", command=self.save, width=12, height=3)
        self.save_button.grid(row=6, column=3)

        self.filename = None

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.predicted_img=None
        self.predicted_img2 = None
        self.old_x = None
        self.old_y = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.eraser_on = False
        self.active_button = self.rect_button
        self.isPainting = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.bind('<Button-1>', self.beginPaint)
        self.c.bind('<Enter>', self.icon2pen)
        self.c.bind('<Leave>', self.icon2mice)
        self.mode = 'poly'
        self.rect_buf = None
        self.line_buf = None
        assert self.mode in ['rect', 'poly']
        self.paint_color = self.MARKER_COLOR
        self.mask_candidate = []
        self.rect_candidate = []
        self.im_h = None
        self.im_w = None
        self.mask = None
        self.result = None
        self.blank = None
        self.line_width = 20

        ##################################################################
        # wny self.model = GMCNNModel()
        self.model=PConvUnet()
        self.model2=PConvUnet()
        self.reuse = False
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = False
        self.sess = tf.Session(config=sess_config)

        self.input_image_tf = tf.placeholder(dtype=tf.float32,
                                             shape=[1, self.config.img_shapes[0], self.config.img_shapes[1], 3])
        self.input_mask_tf = tf.placeholder(dtype=tf.float32,
                                            shape=[1, self.config.img_shapes[0], self.config.img_shapes[1], 1])

        ###############################################################
        # wny
        # output = self.model.evaluate(self.input_image_tf, self.input_mask_tf, config=self.config, reuse=self.reuse)
        # output = (output + 1) * 127.5
        # output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
        # self.output = tf.cast(output, tf.uint8)

        # # load pretrained model
        # vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
        #                       vars_list))
        # self.sess.run(assign_ops)
        #self.model.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\pconv_imagenet.26-1.07.h5", train_bn=False)
        # P1T1
        # self.model.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\weights.07-1.89.h5", train_bn=False)
        # P1T2
        # self.model.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\weights.10-1.74.h5", train_bn=False)
        # P1T3
        self.model.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t152\weights.31-1.20.h5", train_bn=False)
        self.model2.load(r"D:\PycharmProjects2\PConv-Keras\data\logs\Thanka_phase1\p1t169\weights.13-0.98.h5",train_bn=False)
        # wny
        ###############################################################
        print('Model loaded.')

    def checkResp(self):
        assert len(self.mask_candidate) == len(self.rect_candidate)

    def load(self):
        self.filename = tkFileDialog.askopenfilename(initialdir='./imgs',
                                                     title="Select file",
                                                     filetypes=(("all files", "*.*"), ("png files", "*.png"), ("jpg files", "*.jpg")
                                                                ))
        self.filename_ = self.filename.split('/')[-1][:-4]
        self.filepath = '/'.join(self.filename.split('/')[:-1])
        print(self.filename_, self.filepath)
        try:
            photo = Image.open(self.filename)
            self.image = cv2.imread(self.filename)
        except:
            print('do not load image')
        else:
            self.im_w, self.im_h = photo.size
            self.mask= np.zeros((self.im_h, self.im_w, 3)).astype(np.uint8)
            #self.mask=np.zeros_like(self.image)
            print(photo.size)
            self.displayPhoto = photo
            self.displayPhoto = self.displayPhoto.resize((self.im_w, self.im_h))
            self.draw = ImageDraw.Draw(self.displayPhoto)
            self.photo_tk = ImageTk.PhotoImage(image=self.displayPhoto)
            self.c.create_image(0, 0, image=self.photo_tk, anchor=NW)
            self.rect_candidate.clear()
            self.mask_candidate.clear()
            if self.blank is None:
                self.blank = Image.open('imgs/blank.png')
            self.blank = self.blank.resize((self.im_w, self.im_h))
            self.blank_tk = ImageTk.PhotoImage(image=self.blank)
            self.out.create_image(0, 0, image=self.blank_tk, anchor=NW)
            self.out2.create_image(0, 0, image=self.blank_tk, anchor=NW)

    def save(self):
        #img = np.array(self.displayPhoto)
        #cv2.imwrite(os.path.join(self.filepath, 'tmp.png'), img)

        if self.mode == 'rect':
            self.mask[:,:,:] = 0
            for rect in self.mask_candidate:
                self.mask[rect[1]:rect[3], rect[0]:rect[2], :] = 1
        #self.mask=1-self.mask
        cv2.imwrite(os.path.join(self.filepath, self.filename_+'_mask.png'), (1-self.mask)*255)
        #wny cv2.imwrite(os.path.join(self.filepath, self.filename_ + '_gm_result.png'), self.result[0][:, :, ::-1])
        #cv2.imwrite(os.path.join(self.filepath, self.filename_ + '_gm_result.png'), self.predicted_img)
        cv2.imwrite(os.path.join(self.filepath, self.filename_ + '_result.png'), cv2.cvtColor(self.predicted_img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(self.filepath, self.filename_ + '_result2.png'),cv2.cvtColor(self.predicted_img2, cv2.COLOR_BGR2RGB))

    def fill(self):
        if self.mode == 'rect':
            for rect in self.mask_candidate:
                self.mask[rect[1]:rect[3], rect[0]:rect[2], :] = 1
        ########################################################################
        # wny: to create a three-channel mask, input the original single-layer mask's value to each channel of new mask：
        mask_channel=np.zeros_like(self.image)
        mask_channel[:, :, 0] = self.mask[:,:,0]
        mask_channel[:, :, 1] = self.mask[:,:,0]
        mask_channel[:, :, 2] = self.mask[:,:,0]
        self.mask=mask_channel
        # wny: to exchange 0 with 1, 1 with 0 in new mask:
        #self.mask=1-self.mask
        image_temp = Image.open(self.filename)
        image_temp =np.array(image_temp)/255
        # wny: add mask to input image:
        image_temp[self.mask==1]=1
        image = np.expand_dims(image_temp, 0)
        mask = np.expand_dims(1-self.mask, 0)

        ########################################################################
        print(image.shape)
        print(mask.shape)

        image_temp = Image.fromarray(np.uint8(image_temp*255))
        image_temp.save('./imgs/masked_input.png')

        cv2.imwrite('./imgs/mask.png', (1-self.mask)*255)
        #########################################################################
        # wny
        # self.result = self.sess.run(self.output, feed_dict={self.input_image_tf: image * 1.0,
        #                                                     self.input_mask_tf: mask * 1.0})
        # output the predicted image
        self.predicted_img = (self.model.predict([image, mask])[0])*255
        self.predicted_img2 = (self.model2.predict([image, mask])[0]) * 255

        # wny cv2.imwrite('./imgs/tmp.png', self.result[0][:, :, ::-1])
        cv2.imwrite('./imgs/result.png', cv2.cvtColor(self.predicted_img, cv2.COLOR_BGR2RGB))
        cv2.imwrite('./imgs/result2.png', cv2.cvtColor(self.predicted_img2, cv2.COLOR_BGR2RGB))
        ##########################################################################
        photo = Image.open('./imgs/result.png')
        photo2 = Image.open('./imgs/result2.png')
        self.displayPhotoResult = photo
        self.displayPhotoResult2 = photo2
        self.displayPhotoResult = self.displayPhotoResult.resize((self.im_w, self.im_h))
        self.displayPhotoResult2 = self.displayPhotoResult2.resize((self.im_w, self.im_h))
        self.photo_tk_result = ImageTk.PhotoImage(image=self.displayPhotoResult)
        self.photo_tk_result2 = ImageTk.PhotoImage(image=self.displayPhotoResult2)
        self.out.create_image(0, 0, image=self.photo_tk_result, anchor=NW)
        self.out2.create_image(0, 0, image=self.photo_tk_result2, anchor=NW)
        return

    def use_rect(self):
        self.activate_button(self.rect_button)
        self.mode = 'rect'

    def use_poly(self):
        self.activate_button(self.poly_button)
        self.mode = 'poly'

    def revoke(self):
        if len(self.rect_candidate) > 0:
            self.c.delete(self.rect_candidate[-1])
            self.rect_candidate.remove(self.rect_candidate[-1])
            self.mask_candidate.remove(self.mask_candidate[-1])
        self.checkResp()

    def clear(self):
        self.mask = np.zeros((self.im_h, self.im_w, 1)).astype(np.uint8)
        if self.mode == 'poly':
            photo = Image.open(self.filename)
            self.image = cv2.imread(self.filename)
            self.displayPhoto = photo
            self.displayPhoto = self.displayPhoto.resize((self.im_w, self.im_h))
            self.draw = ImageDraw.Draw(self.displayPhoto)
            self.photo_tk = ImageTk.PhotoImage(image=self.displayPhoto)
            self.c.create_image(0, 0, image=self.photo_tk, anchor=NW)
        else:
            if self.rect_candidate is None or len(self.rect_candidate) == 0:
                return
            for item in self.rect_candidate:
                self.c.delete(item)
            self.rect_candidate.clear()
            self.mask_candidate.clear()
            self.checkResp()

    #TODO: reset canvas
    #TODO: undo and redo
    #TODO: draw triangle, rectangle, oval, text

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def beginPaint(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.isPainting = True

    def paint(self, event):
        if self.start_x and self.start_y and self.mode == 'rect':
            self.end_x = max(min(event.x, self.im_w), 0)
            self.end_y = max(min(event.y, self.im_h), 0)
            rect = self.c.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, fill=self.paint_color)
            if self.rect_buf is not None:
                self.c.delete(self.rect_buf)
            self.rect_buf = rect
        elif self.old_x and self.old_y and self.mode == 'poly':
            line = self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                                      width=self.line_width, fill=self.paint_color, capstyle=ROUND,
                                      smooth=True, splinesteps=36)
            cv2.line(self.mask, (self.old_x, self.old_y), (event.x, event.y), (1), self.line_width)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None
        if self.mode == 'rect':
            self.isPainting = False
            rect = self.c.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y,
                                           fill=self.paint_color)
            if self.rect_buf is not None:
                self.c.delete(self.rect_buf)
            self.rect_buf = None
            self.rect_candidate.append(rect)

            x1, y1, x2, y2 = min(self.start_x, self.end_x), min(self.start_y, self.end_y),\
                             max(self.start_x, self.end_x), max(self.start_y, self.end_y)
            # up left corner, low right corner
            self.mask_candidate.append((x1, y1, x2, y2))
            print(self.mask_candidate[-1])

    def icon2pen(self, event):
        return

    def icon2mice(self, event):
        return


if __name__ == '__main__':
    config = TestOptions().parse()
    config.mode = 'silent'
    ge = Paint(config)
