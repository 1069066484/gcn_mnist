import torchvision
from torchvision import transforms
import torch
import numbers
import random
from package.dataset import functional2 as F
from collections import Iterable


class ChannelExp(object):
    def __init__(self, num_chs=3):
        self.num_chs = num_chs

    def __call__(self, single_ch):
        return torch.cat([single_ch for _ in range(self.num_chs)])

    def __repr__(self):
        return self.__class__.__name__ + '_ChannelExp'


class FixRotation(object):
    def __init__(self):
        self.resample = False
        self.expand = False
        self.center = None

    def __call__(self, img):
        rd = random.random()
        if rd < 0.5:
            angle = 0
        elif rd < 0.75:
            angle = 90
        else:
            angle = 270
        return F.rotate(img, angle, self.resample, self.expand, self.center)


class Norm01(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img[img < 0.5] = 0.0
        img[img > 0.01] = 1.0
        return img


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        """
        :param size: int or float
        """
        if isinstance(size, Iterable):
            self.size = size
        else:
            self.size = (size, size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        if isinstance(self.size[0], int):
            size = (int(self.size[0]), int(self.size[1]))
        else:
            size = (int(self.size[0] * img.size[0]), int(self.size[1] * img.size[1]))

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < size[1]:
            img = F.pad(img, (size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < size[0]:
            img = F.pad(img, (0, size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


if __name__=='__main__':
    pass