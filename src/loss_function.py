import math
import torch
import torch.nn as nn

class Loss_function(nn.Module):
    def __init__(self):
        super(Loss_function, self).__init__()

    def IOU_calculator(self, box1, box2):
        # current box size: [confidence, cx, cy, w, h, class]
        # target box size: [top, left, bottom, right]
        # box1
        top1 = box1[2] + 1 / 2 * box1[4]
        bottom1 = box1[2] + 1 / 2 * box1[4]
        left1 = box1[1] - 1 / 2 * box1[3]
        right1 = box1[1] + 1 / 2 * box1[3]
        box1 = [top1, left1, bottom1, right1]
        # box2
        top2 = box2[2] + 1 / 2 * box2[4]
        bottom2 = box2[2] + 1 / 2 * box2[4]
        left2 = box2[1] - 1 / 2 * box2[3]
        right2 = box2[1] + 1 / 2 * box2[3]
        box2 = [top2, left2, bottom2, right2]
        # calculate the IOU
        in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
        in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
        inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
                (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
        iou = inter / union
        return iou

    # define the 5 components of the loss function
    def center_loss(self, box, gt_box):
        # box size: [confidence, cx, cy, w, h]
        loss = 5 * (torch.sum((box[1] - gt_box[1]) ** 2) + torch.sum((box[2] - gt_box[2]) ** 2))
        return loss

    def wh_loss(self, box, gt_box):
        # box size: [confidence, cx, cy, w, h]
        loss = 5 * (torch.sum((math.sqrt(box[3]) - math.sqrt(gt_box[3])) ** 2) + torch.sum(
            (math.sqrt(box[4]) - math.sqrt(gt_box[4])) ** 2))
        return loss

    def object_confidence_loss(self, iou):
        return iou

    def noobject_confidence_loss(self, iou):
        return iou

    def classification_loss(self, pred_class, gt_box):
        loss = torch.sum((pred_class - gt_box[10:]) ** 2)

    # the whole process
    def forward(self, pred, labels):
        # five components of the loss function
        bnd_center_loss = 0  # bounding box center loss
        bnd_wh_loss = 0  # bounding box width and height loss
        obj_confident_loss = 0  # confidence loss for bounding box with object witin
        noobj_confident_loss = 0  # confidence loss for bounding box without object within
        classification_loss = 0  # classification loss
        n_batch = labels.shape[0]
        for i in range(n_batch):  #
            for m in range(7):
                for n in range(7):
                    if labels[i, m, n, 0] == 1:
                        # first we need to select the bounding box which has
                        # a higher IOU with the ground truth bounding box
                        current_pred = pred[i, m, n]
                        current_label = labels[i, m, n]
                        box1 = current_pred[0:5]
                        pred_class = current_pred[10:]
                        box2 = current_pred[5:10]
                        iou1 = self.IOU_calculator(box1, current_label)
                        iou2 = self.IOU_calculator(box2, current_label)
                        print('box1:  {}\n, box2: {}'.format(iou1,iou2))
                        # if box1 has a higher iou, then box1 would be responsible for this task.
                        if iou1 >= iou2:
                            # calculating bounding box center loss
                            bnd_center_loss += self.center_loss(box1, labels[i, m, n, 0:5])
                            # calculating bounding box wh loss
                            bnd_wh_loss += self.wh_loss(box1, labels[i, m, n, 0:5])
                            # calculating object confidence loss
                            obj_confident_loss += self.object_confidence_loss(iou1)
                            # calculating noobject confidence loss
                            noobj_confident_loss += self.noobject_confidence_loss(iou2)
                            # calculating classification loss
                            classification_loss += self.classification_loss(pred_class, labels)
                        # if box2 has a higher iou, then box2 would be responsible for this task.
                        else:
                            # calculating bounding box center loss
                            bnd_center_loss += self.center_loss(box2, labels[i, m, n, 0:5])
                            # calculating bounding box wh loss
                            bnd_wh_loss += self.wh_loss(box2, labels[i, m, n, 0:5])
                            # calculating object confidence loss
                            obj_confident_loss += self.object_confidence_loss(iou2)
                            # calculating noobject confidence loss
                            noobj_confident_loss += self.noobject_confidence_loss(iou1)
                            # calculating classification loss
                            classification_loss += self.classification_loss(pred_class, labels)

