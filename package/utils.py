import logging
import numpy as np
import os
import datetime
import torch
import cv2





exists = os.path.exists
join = os.path.join


def posfix_filename(filename, postfix):
    if not filename.endswith(postfix):
        filename += postfix
    return filename


def mkdir(path):
    if not exists(path):
        os.mkdir(path)
    return path


def npfilename(filename, z=False):
    return posfix_filename(filename, '.npy') if not z else posfix_filename(filename, '.npz')


def pkfilename(filename):
    return posfix_filename(filename, '.pkl')


def csvfilename(filename):
    return posfix_filename(filename, '.csv')


def h5filename(filename):
    return posfix_filename(filename, '.h5')


def logfilename(filename):
    return posfix_filename(filename, '.log')


def txtfilename(filename):
    return posfix_filename(filename, '.txt')


def giffilename(filename):
    return posfix_filename(filename, '.gif')


npfn = npfilename
pkfn = pkfilename
csvfn = csvfilename
h5fn = h5filename
logfn = logfilename
txtfn = txtfilename
giffn = giffilename


def curr_time_str():
    return datetime.datetime.now().strftime('%y%m%d%H%M%S')



def make_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = log_file
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('logfile = {}'.format(logfile))
    return logger


def get_pre_from_matches(matches):
    """
    :param matches: A n-by-m matrix. n is number of test samples, m is the top m elements used for evaluation
    :return: precision
    """
    return np.mean(matches)


def _map_change(inputArr):
    dup = np.copy(inputArr)
    for idx in range(inputArr.shape[1]):
        if idx != 0:
            # dup cannot be bool type
            dup[:,idx] = dup[:,idx-1] + dup[:,idx]
    return np.multiply(dup, inputArr)


def get_map_from_matches(matches):
    """
    mAP's calculation refers to https://github.com/ShivaKrishnaM/ZS-SBIR/blob/master/trainCVAE_pre.py.
    :param matches: A n-by-m matrix. n is number of test samples, m is the top m elements used for evaluation
            matches[i][j] == 1 indicates the j-th retrieved test image j belongs to the same class as test sketch i,
            otherwise, matches[i][j] = 0.
    :return: mAP
    """
    temp = [np.arange(matches.shape[1]) for _ in range(matches.shape[0])]
    mAP_term = 1.0 / (np.stack(temp, axis=0) + 1.0)
    precisions = np.multiply(_map_change(matches), mAP_term)
    mAP = np.mean(precisions, axis=1)
    return np.mean(mAP)


def center_bimg(img, save=False, neg=True):
    """
    Center the contents of images.
    :param img: an np-array image.
    :param save: save the image or not, provided the given image is a string, indicating the path.
    :param neg: whether use negative. Set true if the text is written in black.
    :return: centered image.
    """
    if isinstance(img, str):
        path = img
        img = cv2.imread(path)
    else:
        path = None
    shape = img.shape
    if len(shape) == 3:
        img = img[:, :, 0]
    m = np.max(img)
    img[img < m / 2] = 0
    img[img > m / 2] = m
    if neg:
        img = m - img
    for top in range(img.shape[0]):
        if np.sum(img[top]) != 0:
            break
    for bot in range(img.shape[0] - 1, -1, -1):
        if np.sum(img[bot]) != 0:
            break
    for left in range(img.shape[1]):
        if np.sum(img[:, left]) != 0:
            break
    for right in range(img.shape[1] - 1, -1, -1):
        if np.sum(img[:, right]) != 0:
            break
    if neg:
        img = m - img
    bot += 1
    right += 1
    img = img[top: bot, left: right]
    new_img = np.zeros(shape[:2], dtype=img.dtype) + m * neg
    new_top = (shape[0] - (bot - top)) // 2
    new_bot = new_top + (bot - top)
    new_left = (shape[1] - (right - left)) // 2
    new_right = new_left + (right - left)
    new_img[new_top: new_bot, new_left: new_right] = img
    img = new_img
    if len(shape) == 3:
        img = np.stack([img for _ in range(shape[2])], 2)
    if save and path is not None:
        cv2.imwrite(path, img)
    return img


def cv_show1(arr, sz=400):
    """
    :param arr:  an array. torch.Tensor or numpy array. Ensure arr represents 3-dimension array.
    :param sz: size of the image.
    :return: None.
    """
    if isinstance(sz, int):
        sz = (sz, sz)

    if len(arr.shape) == 4:
        arr = arr[0]
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu()
        arr = np.transpose(arr.numpy(), (1, 2, 0))
        if sz is not None:
            arr = cv2.resize(arr, sz)
    cv2.imshow("cv_show1", arr)
    cv2.waitKeyEx()


def _test_center_bimg():
    img = r'C:\Users\22\Desktop\tmp\img.png'
    img_ = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1.0, 0, 0],
                    [1.0, 1.0, 1.0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    ])
    # img = np.stack([img for _ in range(3)], 2)
    img_new = center_bimg(img)
    img = cv2.imread(img)
    # print(img_new, '\n', img)
    # print(img_new.shape, img.shape, np.max(img))
    # exit()
    print(img_new.shape, img.shape)
    img_new = cv2.resize(img_new, (400, 400))
    img = cv2.resize(img, (400, 400))

    cv2.imshow("centered", img_new)
    cv2.imshow("uncentered", img)
    cv2.waitKey()


if __name__ == '__main__':
    # _test_center_bimg()
    # text2img(txt=u'5', path=r'C:\Users\22\Desktop\tmp\img.png')
    pass



