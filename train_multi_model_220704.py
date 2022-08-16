#
# train Fingerprint alignments network 2022.07104
#
import copy
import os
import pathlib
import sys
import logging
import argparse
import itertools
from datetime import datetime
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torchvision.transforms import Compose
from models_fingerprints import MobileNetv2_input2channel, ShuffleNetV2_input2channel, \
    resnet18_input2channel, resnet50_input2channel, resnet101_input2channel, Googlenet_input2channel, \
    Vgg11_bn_input2channel, densenet121_input2channel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # logging.info("Using CUDA...")


# class TrainAugmentation220623:
#     def __init__(self, size, mean=0, std=1.0):
#         """
#         车牌增强，不需要调整下，以改变不适合的增强方法，@220623
#         Args:
#             size: the size the of final image.
#             mean: mean pixel value per channel.
#         """
#         self.mean = mean
#         self.size = size
#         self.augment = Compose([
#             # ConvertFromInts(),
#             # PhotometricDistort(),
#             # Expand(self.mean),
#             # RandomSampleCrop(),
#             # RandomMirror(),
#             ToPercentCoords(),
#             Resize(self.size),
#             SubtractMeans(self.mean),
#             lambda img, boxes=None, labels=None: (img / std, boxes, labels),
#             ToTensor(),
#         ])
#
#     def __call__(self, img, boxes, labels):
#         """
#
#         Args:
#             img: the output of cv.imread in RGB layout.
#             boxes: boundding boxes in the form of (x1, y1, x2, y2).
#             labels: labels of boxes.
#         """
#         return self.augment(img, boxes, labels)


class FingprintAlignDataset:
    def __init__(self, root, corner_index, is_train=True, train_eval_ratio=0.8):
        self.root = pathlib.Path(root)
        self.istrain = is_train
        self.train_eval_ratio = train_eval_ratio
        self.data = self._read_data()
        self.corner_inx = corner_index
        # self.class_stat = None

    def __getitem__(self, index):
        image_name = self.data[index]
        image_file = os.path.join(self.root, image_name)
        sample = np.load(image_file, allow_pickle=True)
        data_input = sample[0][:, :, [0, 4]]
        ori_images = (data_input.astype(np.double) - 127.5) / 127.5
        ori_images = np.transpose(ori_images, [2, 0, 1])  # torch [C,H,W]

        ori_images = torch.from_numpy(ori_images)
        # WARN 标注的是corner四个点，
        input_patch = torch.from_numpy(sample[2][self.corner_inx, :].reshape(-1))
        # pts1 = torch.from_numpy(pts1)

        # if image.shape[2] == 1:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # else:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = sample
        # boxes = copy.copy(image_info['boxes'])
        # boxes[:, 0] *= 1
        # boxes[:, 1] *= 1
        # boxes[:, 2] *= 1
        # boxes[:, 3] *= 1
        # duplicate labels to prevent corruption of dataset
        # labels = copy.copy(image_name['labels'])
        # if self.transform:
        #     image, boxes, labels = self.transform(image, boxes, labels)
        # if self.target_transform:
        #     boxes, labels = self.target_transform(boxes, labels)
        return ori_images, input_patch

    def _read_data(self):
        # annotation_file = f"{self.root}/LPR_keye_bd40000_sub1000_man_draw_20220415_1814_train_100.csv"
        # if self.istrain:
        #     annotation_file = f"{self.root}/LPR_keye_bd40000_sub1000_man_draw_20220415_1814_train_900.csv"
        # annotation_file = os.path.join(self.root, self.csv_file_path)
        # logging.info(f'loading annotations from: {annotation_file}')
        # annotations = pd.read_csv(annotation_file)
        # logging.info(f'annotations loaded from:  {annotation_file}')
        # class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        # 过滤标记，用于区分train/eval
        # annotations = annotations[annotations['type'] == self.dataset_type]
        # class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = [os.path.basename(p) for p in sorted(glob(os.path.join(self.root, '*')))]
        if self.istrain:
            sample_length = int(len(data) * self.train_eval_ratio)
            data = data[:sample_length]
        else:
            sample_length = int(len(data) * (1 - self.train_eval_ratio))
            data = data[:sample_length]

        # for sample_path in sorted(glob(os.path.join(self.root, '*'))):
        #     img_path = os.path.join(self.root, image_id)
        #     if os.path.isfile(img_path) is False:
        #         logging.error(f'missing ImageID {image_id}.jpg - dropping from annotations')
        #         continue
        #     boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
        #     # make labels 64 bits to satisfy the cross_entropy function
        #     labels = np.array([class_dict[name] for name in group["ClassName"]], dtype='int64')
        #     # print('found image {:s}  ({:d})'.format(img_path, len(data)))
        #     data.append({
        #         'image_id': image_id,
        #         'boxes': boxes,
        #         'labels': labels
        #     })
        print('num images:  {:d}'.format(len(data)))
        return data

    def __len__(self):
        return len(self.data)

    # def __repr__(self):
    #     if self.class_stat is None:
    #         self.class_stat = {name: 0 for name in self.class_names[1:]}
    #         for example in self.data:
    #             for class_index in example['labels']:
    #                 class_name = self.class_names[class_index]
    #                 self.class_stat[class_name] += 1
    #     content = ["Dataset Summary:"
    #                f"Number of Images: {len(self.data)}",
    #                f"Minimum Number of Images for a Class: {self.min_image_num}",
    #                "Label Distribution:"]
    #     for class_name, num in self.class_stat.items():
    #         content.append(f"\t{class_name}: {num}")
    #     return "\n".join(content)

    def _read_sample(self, image_id):
        '''
        sample[0][:,:,0]整指纹
        sample[0][:,:,4]局部指纹
        :param image_id:
        :return:
        '''
        image_file = os.path.join(self.root, image_id)
        sample = np.load(image_file, allow_pickle=True)
        # if image.shape[2] == 1:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # else:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return sample

    # def _balance_data(self):
    #     logging.info('balancing data')
    #     label_image_indexes = [set() for _ in range(len(self.class_names))]
    #     for i, image in enumerate(self.data):
    #         for label_id in image['labels']:
    #             label_image_indexes[label_id].add(i)
    #     label_stat = [len(s) for s in label_image_indexes]
    #     self.min_image_num = min(label_stat[1:])
    #     sample_image_indexes = set()
    #     for image_indexes in label_image_indexes[1:]:
    #         image_indexes = np.array(list(image_indexes))
    #         sub = np.random.permutation(image_indexes)[:self.min_image_num]
    #         sample_image_indexes.update(sub)
    #     sample_data = [self.data[i] for i in sample_image_indexes]
    #     return sample_data


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, labels = data  # labels 可能值，4，6，8
        images = images.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor)  # .to(device)  # .double()

        optimizer.zero_grad()
        locations = net(images)
        # sh_box = boxes.shape
        # sh_loc = locations.shape
        # print(sh_box, sh_loc, confidence.shape)

        regression_loss = criterion(
            locations.cpu(),
            labels,
        )
        loss = regression_loss
        loss.backward()
        optimizer.step()
        # print('running_loss ', running_loss, loss.item(), regression_loss, classification_loss)
        running_loss += loss.detach().cpu().item()
        running_regression_loss += regression_loss.item()
        # running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}/{len(loader)}, " +
                f"Avg Loss: {avg_loss:.4f}, " +
                f"Avg Regression Loss {avg_reg_loss:.4f}, " +
                f"Avg Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, labels = data  # labels 可能值，4，6，8
        images = images.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)  # .double()

        num += 1

        with torch.no_grad():
            locations = net(images)
            regression_loss = criterion(locations,
                                        labels)
            loss = regression_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        # running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    '''
    note0708 经过反复测试，训练过程中会崩溃，原因不明，RuntimeError: CUDA error: unspecified launch failure，
            而且只在resnet网络中，考虑先不训练resnet，只训练mobilenet,sufflenet
            MobileNetV3_input2channel
            resnet18_input2channel
            Vgg11_bn_input2channel
            以上网络不能训练
    
    '''
    paths_4_dataset = glob('dataset_train*edge*/')
    models_prototypes = [
        MobileNetv2_input2channel,
        # MobileNetV3_input2channel,
        ShuffleNetV2_input2channel,
        resnet18_input2channel,
        # resnet50_input2channel,
        # resnet101_input2channel,
        densenet121_input2channel,
        Googlenet_input2channel,
        Vgg11_bn_input2channel
    ]
    gt_corners = [[0, 1], [0, 2], [0, 1, 2], [0, 1, 2, 3]]
    # gt_corners = [[0, 1, 2, 3]]
    for train_instance in itertools.product(paths_4_dataset, models_prototypes, gt_corners):
        ds = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.cuda.empty_cache()
        # timer = Timer()
        # iter model
        create_net = train_instance[1]
        checkpoint_folder = 'chk_%s_%s' % (create_net.__name__, ds)
        batch_size = 32
        gt_corners_index = train_instance[2]
        learning_rate = 0.001
        num_epochs = 60
        num_workers = 8
        validation_epochs = 1
        image_std = 1
        dataset_path = train_instance[0]

        try:
            # py39 有force参数指定可能强制除去之前的handler，这里使用兼容写法，0708
            logging.getLogger().removeHandler(logging.getLogger().handlers[0])
            logging.getLogger().removeHandler(logging.getLogger().handlers[0])
        except:
            pass
        logging.basicConfig(
            # force=
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler('log_%s_%s_%s.log' % (ds,
                                                          create_net,
                                                          os.path.basename(dataset_path))),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(train_instance)
        logging.warning(
            'start training. model: %s path %s date %s corner %s' % (create_net, dataset_path, ds, gt_corners_index))
        num_classes = len(gt_corners_index) * 2  # 2维坐标点

        # make sure that the checkpoint output dir exists
        if checkpoint_folder:
            checkpoint_folder = os.path.expanduser(checkpoint_folder)
            if not os.path.exists(checkpoint_folder):
                os.mkdir(checkpoint_folder)
        # create data transforms for train/test/val
        # train_transform = TrainAugmentation220623(128, 255 / 2, image_std)
        # test_transform = TestTransform(128, 255 / 2, image_std)

        # load datasets (could be multiple)
        logging.info("Prepare training datasets.")
        # datasets = []
        ##
        dataset = FingprintAlignDataset(dataset_path, corner_index=gt_corners_index)
        label_file = os.path.join(checkpoint_folder, "labels.txt")
        # store_labels(label_file, dataset.class_names)
        ##
        # create training dataset
        logging.info(f"Stored labels into file {label_file}.")
        train_dataset = dataset
        logging.info("Train dataset size: {}".format(len(train_dataset)))
        train_loader = DataLoader(train_dataset, batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)

        # create validation dataset
        logging.info("Prepare Validation datasets.")
        val_dataset = FingprintAlignDataset(dataset_path,
                                            corner_index=gt_corners_index,
                                            is_train=False)
        logging.info("Validation dataset size: {}".format(len(val_dataset)))

        val_loader = DataLoader(val_dataset, batch_size,
                                num_workers=num_workers,
                                shuffle=False)

        # create the network
        logging.info("Build network.")
        cur_model = create_net(num_classes)
        min_loss = -10000.0
        last_epoch = -1
        # move the model to GPU
        cur_model.to(DEVICE)
        # define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(cur_model.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9,
                                    )
        # optimizer = torch.optim.Adam(cur_model.parameters(),
        #                             lr=learning_rate,
        #                             )
        logging.info(f"Learning rate: {learning_rate}")

        # train for the desired number of epochs
        logging.info(f"Start training from epoch {last_epoch + 1}.")

        for epoch in range(last_epoch + 1, num_epochs):
            # scheduler.step()
            train(train_loader, cur_model, criterion, optimizer, debug_steps=10,
                  device=DEVICE, epoch=epoch)

            if epoch % validation_epochs == 0 or epoch == num_epochs - 1:
                val_loss, val_regression_loss, val_classification_loss = test(val_loader, cur_model, criterion, DEVICE)
                logging.info(
                    f"Epoch: {epoch}, " +
                    f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Regression Loss {val_regression_loss:.4f}, " +
                    f"Validation Classification Loss: {val_classification_loss:.4f}"
                )
                if epoch == num_epochs - 1:
                    model_path = os.path.join(checkpoint_folder,
                                              f"{create_net}-Epoch-{epoch}-Loss-{val_loss}.pth")
                    # cur_model.save(model_path)
                    torch.save(cur_model.state_dict(), model_path)
                    logging.info(f"Saved model {model_path}")

        logging.info("Task done, exiting program.")
