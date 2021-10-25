from torch.utils.data import Dataset

from data_preprocessing import data_preprocessing
import torchvision


class mask_data_set(Dataset):
    def __init__(self, is_train=True):
        if is_train:
            self.dataset = data_preprocessing(r'C:\Users\17391\PycharmProjects\yolo_v1\images_train',
                                              r'C:\Users\17391\PycharmProjects\yolo_v1\annotations_train')
            self.image, self.label = self.dataset.one_stop()
        else:
            self.dataset = data_preprocessing(r'C:\Users\17391\PycharmProjects\yolo_v1\images_val',
                                         r'C:\Users\17391\PycharmProjects\yolo_v1\annotations_val')
            self.image, self.label = self.dataset.one_stop()

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        return torchvision.transforms.ToTensor()(self.image[item]), torchvision.transforms.ToTensor()(self.label[item])
