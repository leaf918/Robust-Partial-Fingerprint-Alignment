import logging
import os.path
import sys
import time
from datetime import datetime
from glob import glob
from multiprocessing import Pool

import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EuclideanTransform
import matplotlib.pyplot as plt
from skimage.transform import matrix_transform
import cv2
from tqdm import tqdm

# Find sparse feature correspondences between left and right image.
from sklearn.metrics import mean_squared_error


def estimate_transform_in2fingerprint(img0, img1, patch_edge_half, ax, aximg):
    descriptor_extractor = ORB()

    descriptor_extractor.detect_and_extract(img0)
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img1)
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors_left,
                                descriptors_right,
                                cross_check=True)

    print(f'Number of matches: {matches.shape[0]}')

    # Estimate the epipolar geometry between the left and right image.
    # ransac requires ONLY 2D coordinates.
    tform, inliers = ransac((keypoints_left[matches[:, 0]],
                             keypoints_right[matches[:, 1]]),
                            EuclideanTransform,
                            min_samples=2,
                            residual_threshold=1,
                            max_trials=5000,
                            )

    plt.gray()
    plt.subplot(3, 1, 1)
    plot_matches(plt, img0, img1, keypoints_left, keypoints_right, matches, only_matches=True)
    # plt.axis('off')
    # plt.set_title("match full _ patial")
    # plt.show()

    # matched coordinates in two images.
    imsize = (128, 128)
    imgori = img0.copy()
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
    # rect_a_out = matrix_transform(rect_a, np.linalg.inv(tform.params)).astype(np.int)
    rect_a_out = matrix_transform(rect_a, tform.params).astype(np.int)
    # imgori = cv2.circle(imgori, (rect_a_out[0, 0], rect_a_out[0, 1]), 10, 255, -1)
    # imgori = cv2.circle(imgori, (rect_a_out[1, 0], rect_a_out[1, 1]), 10, 255, -1)
    # imgori = cv2.circle(imgori, (rect_a_out[2, 0], rect_a_out[2, 1]), 10, 255, -1)
    # imgori = cv2.circle(imgori, (rect_a_out[3, 0], rect_a_out[3, 1]), 10, 255, -1)
    plt.subplot(3, 1, 2)

    imgori = cv2.polylines(imgori, [rect_a_out], True, (255, 255, 255), 3)
    drawed_img = cv2.polylines(aximg.copy(), [rect_a_out], True, (211, 211, 211), 3)

    plt.imshow(imgori)
    plt.subplot(3, 1, 3)
    plt.imshow(drawed_img)

    plt.show()

    inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
    inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

    print("matrix details %s" % tform.params,
          # model.scale,
          # model.translation,
          # model.rotation,
          )
    return tform, rect_a_out


# def run(path):
#     try:
#         edge = os.path.split(path)[0]
#         edge = int(edge[edge.find('edge') + 4:])
#         pimg = np.load(path, allow_pickle=True)
#         tform, edge40_corner = estimate_transform_in2fingerprint(pimg[0][:, :, 0],
#                                                                  pimg[0][:, :, 4],
#                                                                  patch_edge_half=edge,
#                                                                  ax=None)
#         mse = mean_squared_error(pimg[2], edge40_corner)
#         # print('mse of ORB_ransac:%s' % mse)
#         # logging.info('ORB  %s' % (pp))
#
#         logging.info('ORB tform %s %s edge[%s] mse[%s] ' % (path, tform, edge, mse))
#     except:
#         return
#
#     # logging.info('ORB MSE %s %s' % (ii, mse))
#     # mses.append(mse)
#     # logging.info('total ORB MSE %s' % (np.mean(mse)))


def run_one(i=0):
    try:
        path = glob('dataset_train_liqiang_20220701_071347_num8000_edge40/*')[i]
        # edge = os.path.split(path)[0]
        edge = 20
        pimg = np.load(path, allow_pickle=True)
        # plt.imshow(pimg[0][:, :, 1]), plt.show()

        tform, edge40_corner = estimate_transform_in2fingerprint(pimg[0][:, :, 0],
                                                                 pimg[0][:, :, 4],
                                                                 patch_edge_half=edge,
                                                                 ax=None,
                                                                 aximg=pimg[0][:, :, 1])
        mse = mean_squared_error(pimg[2], edge40_corner)
        # print('mse of ORB_ransac:%s' % mse)
        # logging.info('ORB  %s' % (pp))

        logging.info('ORB tform %s %s edge[%s] mse[%s] ' % (path, tform, edge, mse))

    except Exception as e:
        return


if __name__ == '__main__':
    for ii in range(10):
        run_one(ii)
    # batch run.
    # paths_4_dataset = glob('dataset_train*/')
    # mses = []
    # for ii, pp in tqdm(enumerate(paths_4_dataset)):
    #     ds = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     try:
    #         # py39 有force参数指定可能强制除去之前的handler，这里使用兼容写法，0708
    #         logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    #         logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    #     except:
    #         pass
    #     logging.basicConfig(
    #         # force=
    #         level=logging.INFO,
    #         format="%(asctime)s [%(levelname)s] %(message)s",
    #         handlers=[
    #             logging.FileHandler('log_%s_ORB.log' % (ds)),
    #             logging.StreamHandler(sys.stdout)
    #         ]
    #     )
    #
    #     pool = Pool(8)  # 创建拥有3个进程数量的进程池
    #     pool.map(run, glob(os.path.join(pp, '*'))[:2000])
    #     pool.close()  # 关闭进程池，不再接受新的进程
    #     pool.join()  # 主进程阻塞等待子进程的退出
