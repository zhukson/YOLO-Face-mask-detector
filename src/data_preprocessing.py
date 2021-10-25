import cv2
import numpy as np
import os
from xml.etree import cElementTree as ET

import torchvision
import matplotlib.pyplot as plt


class data_preprocessing:
    def __init__(self, data_path, annotation_path, cls=3, target_image_size=448, cell_size=7):
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.cell_size = cell_size  # the number of cells we want to split an image into
        self.target_image_size = target_image_size
        self.num_cls = cls

    def load_image(self, image_path):
        img = cv2.imread(image_path)  # read image
        img = cv2.resize(img, (self.target_image_size, self.target_image_size))
        return img/255

    def image_prepare(self):
        data = os.listdir(self.data_path)
        data_set = []
        for filename in data:
            img = self.load_image(os.path.join(self.data_path, filename))
            data_set.append(img)
        return data_set

    def load_annotation(self, annotation_obj):
        data = np.array(os.listdir(self.data_path))
        original_data_size = cv2.imread(os.path.join(self.data_path, data[0])).shape
        height_ratio = original_data_size[0] / self.target_image_size
        width_ratio = original_data_size[1] / self.target_image_size
        # the label for training, the shape is (cell_size, cell_size, )
        label = np.zeros((self.cell_size, self.cell_size,
                          5 + self.num_cls))  # 1 confidence value + 4 position info + number of classes
        annotation_path = os.path.join(self.annotation_path, annotation_obj)
        tree = ET.parse(annotation_path)
        objs = tree.findall('object')
        for obj in objs:
            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text)
            x_max = float(bbox.find('xmax').text)
            y_min = float(bbox.find('ymin').text)
            y_max = float(bbox.find('ymax').text)
            # calculate the center of this image
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            # to determine which grid cell the center point locate at
            x_cell = int(x_center * self.cell_size / self.target_image_size)
            y_cell = int(y_center * self.cell_size / self.target_image_size)
            # if the center is out of bound, set the limit
            if x_cell > 6:
                x_cell = 6
            if y_cell > 6:
                y_cell = 6
            if label[x_cell, y_cell, 0] == 1:
                continue
            label[x_cell, y_cell, 0] = 1
            bndbox_new = [x_center, y_center, width, height]
            label[x_cell, y_cell, 1:5] = bndbox_new
            # we have three classes in total, with_mask, without_mask and mask_weared_incorrect.
            cls2index = {'with_mask': 1, 'without_mask': 2,
                         'mask_weared_incorrect': 3}
            # find the according class for the bounding box
            cls_info = cls2index[obj.find('name').text.lower().strip()]
            label[x_cell, y_cell, 5 + cls_info - 1] = 1
        return label

    def annotation_prepare(self):
        annotation_list = os.listdir(self.annotation_path)
        labels = []
        for annotation_obj in annotation_list:
            label = self.load_annotation(annotation_obj)
            labels.append(label)
        return np.array(labels)

    def one_stop(self):
        data = np.array(self.image_prepare())
        labels = np.array(self.annotation_prepare())
        print('shape of the data :{}'.format(data.shape))
        print('shape of the label :{}'.format(labels.shape))
        return data, labels

    # def draw_box(self, img_name, labels):
    #     img = cv2.imread(self.data_path + "JPEGImages/" + img_name)
    #     h, w = img.shape[:2]
    #     for label in labels:
    #         label = label.split(' ')
    #         label = [float(x.strip()) for x in label]
    #         pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
    #         pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
    #         cv2.putText(img, CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    #         cv2.rectangle(img, pt1, pt2, (0, 0, 255, 2))
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)

aa = data_preprocessing(r'C:\Users\17391\PycharmProjects\yolo_v1\images_train', r'C:\Users\17391\PycharmProjects\yolo_v1\annotations_train')
data_set = aa.one_stop()