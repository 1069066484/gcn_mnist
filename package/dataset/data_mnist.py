import torchvision
from torchvision import transforms
from torch.utils.data import Dataset as torchDataset
from package.dataset.utils import ChannelExp, normalize
from package.utils import *
import numpy as np
import torch


DOWNLOAD_MNIST = not (os.path.exists('./mnist/')) or not os.listdir('./mnist/')


def _mnist(train=True):
    return torchvision.datasets.MNIST(
        root=join(os.path.split(os.path.realpath(__file__))[0], 'mnist_dataset') ,
        train=train,
        download=DOWNLOAD_MNIST,
    )


mnist = _mnist


class Mnist(torchDataset):
    def __init__(self, train=True, doaug=False, exp3ch=False, out_sz=None):
        super(Mnist, self).__init__()
        self.dataset = _mnist(train)
        self.doaug = doaug
        self.exp3ch = exp3ch
        # self.dataset.data = self.dataset.data.reshape([len(self.dataset.data), 28**2, -1])
        # self.dataset.data = torch.from_numpy(self.dataset.data)
        self.dataset.targets = self.dataset.targets.reshape(-1)
        # self.dataset.targets = torch.from_numpy(self.dataset.targets)
        self.ori_sz = self.dataset.data.shape[1]
        self.dataset.data = self.dataset.data.float()
        self.dataset.data = self.dataset.data / torch.max(self.dataset.data)
        self.out_sz = self.ori_sz if out_sz is None else out_sz
        self._build_trans()

    def _build_trans(self):
        trans = [transforms.ToPILImage()]
        if self.doaug:
            trans.append(transforms.RandomRotation(10))
            trans.append(transforms.RandomCrop(int(self.ori_sz * 0.85)))
        trans.append(transforms.Resize(self.out_sz))
        if self.exp3ch:
            trans.append(ChannelExp())
            trans.append(normalize)
        trans.append(transforms.ToTensor())
        self.trans = transforms.Compose(trans)

    def traverse(self, batch_size=16):
        start = 0
        l = self.dataset.data.shape[0]
        while True:
            m = start + batch_size
            m = min(l, m)
            if m <= start:
                break
            yield self.dataset.data[start:m], self.dataset.targets[start:m]
            start += batch_size

    def __getitem__(self, i):
        return self.dataset.data[i], self.dataset.targets[i]

    def __len__(self):
        return self.dataset.data.shape[0]


def _cv2_show_img(data):
    print(data.train_labels.shape, data.data.shape)
    print(data.targets.shape, data.test_data.shape)
    for i in range(0, len(data.data), len(data.data) // 10):
        cv2.imshow('{}, {}'.format(i, data.train_labels[i]), cv2.resize(data.data[i], (224, 224)))
        cv2.waitKeyEx()


def _test_show():
    mnist = Mnist(out_sz=100)
    for img, label_single in mnist:
        print(img.shape, label_single)
        cv2.imshow("1", np.transpose(img.numpy(), (1,2,0)))
        cv2.waitKeyEx()


def _test_conv():
    mnist = Mnist(out_sz=28, conv=True)
    for img, label_single, label_cond in mnist:
        img = np.transpose(img.numpy(), (1, 2, 0))
        label_cond = np.transpose(label_cond.numpy(), (1, 2, 0))
        img = cv2.resize(img, (400, 400))
        label_cond = cv2.resize(label_cond, (400, 400))
        # print(img.shape, label_cond.shape)
        cv2.imshow("1", img)
        cv2.imshow("2", label_cond)
        cv2.waitKeyEx()


if __name__ == '__main__':
    _test_show()

