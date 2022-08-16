import cv2, glob, os
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries
from skimage.morphology import square
from skimage.morphology import closing
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import itertools
import cv2
import tqdm
import random
from numpy.linalg import inv
import imgaug.augmenters as iaa
from skimage import transform
from skimage import img_as_float
import scipy.ndimage as ndimage
from scipy.ndimage import affine_transform
from skimage.transform import matrix_transform
from random import randrange
from random import uniform
from skimage.draw import polygon
import skimage.draw as draw


def ImagePreProcessing2(image_path, patch_edge_half, imsize):
    img = cv2.imread(image_path, 0)[5:-5, 2:-10]
    img = cv2.resize(img, imsize)
    img = np.invert(img)
    rand_rotation = (uniform(-np.pi / 4, np.pi / 4))
    rand_translation = (randrange(-1, +1), randrange(-1, +1))
    tform = transform.SimilarityTransform(
        scale=1,
        rotation=rand_rotation,  # 一个PI对应角度制为180度
        translation=rand_translation)
    tf_img = transform.warp(img, tform.inverse)
    tf_img_inv = transform.warp(tf_img, tform)
    # 四角点顺序，左上开始顺时针方向
    #  这里截局部图，所以使用中心的Rect,不使用整图的触点坐标
    # step :生成原图的warp,warp上画固定rect,保存出patch，映射到原图rect,
    half_w = int(imsize[1] / 2)
    half_h = int(imsize[0] / 2)
    # lt hw-100,hh-100
    # patch_edge = 40
    rect_a = np.array([
        [half_w - patch_edge_half, half_h - patch_edge_half],
        [half_w + patch_edge_half, half_h - patch_edge_half],
        [half_w + patch_edge_half, half_h + patch_edge_half],
        [half_w - patch_edge_half, half_h + patch_edge_half],
    ])
    # pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    # 矩阵求反，再映射坐标
    rect_a_out = matrix_transform(rect_a, np.linalg.inv(tform.params)).astype(np.int)
    # print('pts~rect\n%s\n%s' % (pts, rect_a_out))
    # img = cv2.polylines(img, [pts], True, (255, 255, 255), 3)
    img = cv2.polylines(img, [rect_a_out], True, (255, 255, 255), 3)
    p_ori = np.array([[0, 0],
                      [imsize[0], 0],
                      [imsize[0], imsize[1]],
                      [0, imsize[1]]])
    p_out = matrix_transform(p_ori, tform.params).astype(np.int)
    tf_img = (tf_img * 255).astype(np.uint8)
    # print('----', rect_a[0], rect_a)
    #     tf_img[edge_rect_200[0],edge_rect_200[1]]=255 # draw edge to white color
    tf_img_crop = np.zeros_like(img)
    tf_img_crop[:patch_edge_half * 2, :patch_edge_half * 2] = tf_img[rect_a[0][1]:rect_a[2][1],
                                                              rect_a[0][0]:rect_a[2][0]]

    tf_img = cv2.rectangle(tf_img,
                           rect_a[0],
                           rect_a[2],
                           (255, 255, 255), 2)
    tf_img_inv = (tf_img_inv * 255).astype(np.uint8)
    # 打印变换之后角点
    #     tf_img=cv2.circle(tf_img, (p_out[0,0], p_out[0,1]), 25, 255,-1)
    #     tf_img=cv2.circle(tf_img, (p_out[1,0], p_out[1,1]), 25, 255,-1)
    #     tf_img=cv2.circle(tf_img, (p_out[2,0], p_out[2,1]), 25, 255,-1)
    #     tf_img=cv2.circle(tf_img, (p_out[3,0], p_out[3,1]), 25, 255,-1)
    training_image = np.dstack((img,
                                tf_img,
                                tf_img_inv,
                                tf_img_crop
                                ))
    datum = (training_image,
             p_ori,
             p_out)
    return datum


dataset_size = 1
train_img_size = (128, 128)
filenames = glob.glob("*.BMP")
for i in tqdm.tqdm(range(dataset_size)):
    image_path = random.choice(filenames)
    pimg = ImagePreProcessing2(image_path,  25, train_img_size)
    display_img = np.hstack((pimg[0][:, :, 0],
                             # pimg[0][:, :, 1],
                             # pimg[0][:, :, 2],
                             pimg[0][:, :, 3]
                             ))
    plt.imshow(display_img), plt.show()
