#################
#load data
#################
import numpy as np
import pickle
import cv2

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
#################
# Cifar10 dataset
#################
# train_dir = '/home/zhangxingpeng/inception5-tensorflow/data/cifar-10-batches-py/data_batch_'
# test_dir = '/home/zhangxingpeng/inception5-tensorflow/data/cifar-10-batches-py/test_batch'
# NUM_CLASS = 10
# NUM_TRAIN_BATCH = 5
#################
# Cifar100 dataset
#################
# train_dir = '/raid/data/zhangxingpeng/cifar-100-python/train'
# test_dir = '/raid/data/zhangxingpeng/cifar-100-python/test'
# NUM_CLASS = 100
# NUM_TRAIN_BATCH = 1
#################
# ImageNet32 dataset
# #################
train_dir = '/raid/data/zhangxingpeng/imageNet32/train_data_batch_'
test_dir = '/raid/data/zhangxingpeng/imageNet32/val_data'
NUM_CLASS = 1000
NUM_TRAIN_BATCH = 10


def _read_one_batch(path):
    '''
    This function takes the directory of one batch of data and returns the images and corresponding labels as numpy arrays
    :param path: the directory of one batch of data
    :return: image numpy arrays and label numpy arrays
    '''
    fo = open(path, 'rb')
    dicts = pickle.load(fo,encoding='iso-8859-1')
    fo.close()
    data = dicts['data']
    #label = np.array(dicts['labels'])
    # if dataset is CIFAR100, there are two kinds of label, i.e. 'label' and 'fine_labels'
    #label = np.array(dicts['fine_labels'])
    label = np.array(dicts['labels'])
    label = label - 1

    return data, label

def read_in_all_images(address_list):
    """
    This function reads all training or validation data, shuffles them if needed, and returns the images and the corresponding labels as numpy arrays
    :param address_list: a list of paths of pickle files
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images, image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])

    for address in address_list:
        print('Reading images from ' + address)
        batch_data, batch_label = _read_one_batch(address)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    # This reshape order is really important.
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    data = data.astype(np.float32)

    return data, label

def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i,...] = (image_np[i, ...] - mean) / std

    return image_np

def get_train_data():
    '''
    Read all the train data into numpy array
    :return: all the train data and corresponding labels
    '''
    path_list = []
    for i in range(1, NUM_TRAIN_BATCH+1):
        path_list.append(train_dir + str(i))

    data, label = read_in_all_images(path_list)
    #data, label = read_in_all_images([train_dir])
    #data = whitening_image(data)

    return data, label

def get_valid_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''
    valid_array, valid_labels = read_in_all_images([test_dir])
    #valid_array = whitening_image(valid_array)

    return valid_array, valid_labels