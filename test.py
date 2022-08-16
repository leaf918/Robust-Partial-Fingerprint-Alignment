from glob import glob

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from models_fingerprints import MobileNetv2_input2channel

m_path = "chk_MobileNetv2_input2channel_20220718_131337/<class 'models_fingerprints.MobileNetv2_input2channel'>-Epoch-59-Loss-6.946096887588501.pth"
cur_model = MobileNetv2_input2channel(8)  # .to(device)
cur_model.load_state_dict(torch.load(m_path))
cur_model.eval()
##
# edge = os.path.split(path)[0]
edge = 20
##
fig = plt.figure(figsize=(18, 8))
columns = 4
rows = 3
for i in tqdm(range(1, columns*rows +1)):
    path = glob('dataset_train_liqiang_20220701_071347_num8000_edge40/*')[i]

    sample = np.load(path, allow_pickle=True)

    fig.add_subplot(rows, columns, i)

    data_input = sample[0][:, :, [0, 4]]
    ori_images = (data_input.astype(np.double) - 127.5) / 127.5
    ori_images = np.transpose(ori_images, [2, 0, 1])  # torch [C,H,W]

    ori_images = torch.from_numpy(ori_images.reshape([1, 2, 128, 128]))
    input_patch = torch.from_numpy(sample[2].reshape(-1))
    locations = cur_model(ori_images.type(torch.FloatTensor))  # .to(torch.device("cuda:0")))
    pred = locations.detach().numpy().reshape(-1, 2).astype(np.int64)
    gt_draw = sample[0][:, :, [1]].reshape(128, 128).copy()
    gt_draw = np.dstack([gt_draw, gt_draw, gt_draw])
    # plt.gray()

    # plt.imshow(gt_draw), plt.show()
    img2 = cv2.polylines(gt_draw, [pred], True, (255, 1, 1), 1)
    # img2 = cv2.circle(img2, (pred[0, 0], pred[0, 1]), 10, (255, 1, 1), -1)
    plt.imshow(img2)#, plt.show()

plt.show()
