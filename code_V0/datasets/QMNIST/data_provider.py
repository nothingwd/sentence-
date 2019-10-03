#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-29 04:15
# @Author  : Sijia Chen (sjia.chen@mail.utoronto.ca)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0

''' python inherent libs '''
import os
from collections import defaultdict
import collections

''' third parts libs '''
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

''' custom libs '''

FLAGS = tf.app.flags.FLAGS

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])



colors_digial_map = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'purple': 4, 'saddlebrown': 5, 'brown': 6, 'aqua': 7}
digital_colors_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple', 5: 'saddlebrown', 6: 'brown', 7: 'aqua'}
colors_digial_map = colors_digial_map


class DataSet(object):
  """Container class for a dataset
  """
  def __init__(self,
               images,  # [total_labels, h, w, 3]
               labels, # should contain [total_labels, 6]
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=False,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate(
          (images_rest_part, images_new_part), axis=0), np.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def decode_labels_file(labels_file_path, num_channels):
    ''' 
        each line in the labels file should be img_number, digital, ymin, xmin, ymax, xman, 

        return:
            a list, [img_number, digital, [ymin, xmin, ymax, xman]]
    '''
    with open(labels_file_path, 'r') as f:
        read_labels_data = f.readlines()

    global_labels_data = defaultdict(list)
    for line_label_dt in read_labels_data:
        splitd_line_label = line_label_dt.rstrip('\n').strip().split(',')
        img_number = splitd_line_label[0]
        digital_num = int(splitd_line_label[1])
        if num_channels == 1:
            digital_bbox = list(map(lambda x: int(x), splitd_line_label[2:]))
            global_labels_data[img_number].append([digital_num, *digital_bbox])
        else: 
            digital_color = colors_digial_map[splitd_line_label[2]]
            digital_bbox = list(map(lambda x: int(x), splitd_line_label[3:]))
            global_labels_data[img_number].append([digital_num, digital_color, *digital_bbox])

    return global_labels_data

def extract_image(img_f_path, num_channels):

    if num_channels == 1:
        im = cv2.imread(img_f_path, cv2.IMREAD_GRAYSCALE) # h x w with shape uint32
    else:
        im = cv2.imread(img_f_path, cv2.IMREAD_COLOR) # h x w x 3 with shape uint32

    if im.ndim == 2: # we add 1 as channel in the third axis
        im = np.expand_dims(im, axis=2)
    return im

def align_img_label(imgs_f, global_labels_data):
    used_imgs = list()
    used_labels = list()
    for img_f_name in imgs_f:
        img_number = img_f_name.strip().split('.')[0]
        img_corres_labels = global_labels_data[img_number]
        for  img_label in img_corres_labels:
            used_imgs.append(img_f_name)
            used_labels.append(img_label)

    return used_imgs, used_labels

def integrate_align_img_label(imgs_f, global_labels_data):
    used_imgs = list()
    used_labels = list()
    for img_f_name in imgs_f:
        img_number = img_f_name.strip().split('.')[0]
        img_corres_labels = global_labels_data[img_number]
        img_corres_labels_array = np.array(img_corres_labels)
        used_imgs.append(img_f_name)
        used_labels.append(img_corres_labels)        
        # for  img_label in img_corres_labels:
        #     used_imgs.append(img_f_name)
        #     used_labels.append(img_label)

    return used_imgs, used_labels

def prepare_data(data_path, train_size, num_channels, is_save=True):
    ''' read the data and divide all the data into three categories 

        Args:
            data_path: the str, where the data is stored
                        there must contain images, labels directories
            train_size: size of the training data
            validation_size
        Note: test_size is the total number of images minus train_size
    '''

    raw_images_dir_pt = os.path.join(data_path, 'images')
    raw_labels_dt_pt = os.path.join(data_path, 'labels.txt')

    all_imgs_file = [f for f in os.listdir(raw_images_dir_pt) if '.png' in f]

    global_labels_data = decode_labels_file(raw_labels_dt_pt, num_channels)

    used_imgs_f_name, used_labels = integrate_align_img_label(all_imgs_file, global_labels_data)

    train_imgs_f_name = used_imgs_f_name[:train_size]
    train_labels = used_labels[:train_size]
    test_imgs_f_name = used_imgs_f_name[train_size:]
    test_labels = used_labels[train_size:]


    train_images_data = [extract_image(os.path.join(raw_images_dir_pt, img_f), num_channels) \
                        for img_f in train_imgs_f_name]
    test_images_data = [extract_image(os.path.join(raw_images_dir_pt, img_f), num_channels) \
                        for img_f in test_imgs_f_name]

    train_images_data = np.array(train_images_data)
    train_labels = np.array(train_labels)

    test_images_data = np.array(test_images_data)
    test_labels = np.array(test_labels)

    if is_save: # save the integrated data in the current input data_path
        np.save(os.path.join(data_path, 'train_images_data.npy'), train_images_data)
        np.save(os.path.join(data_path, 'train_labels.npy'), train_labels)
        np.save(os.path.join(data_path, 'test_images_data.npy'), test_images_data)
        np.save(os.path.join(data_path, 'test_labels.npy'), test_labels)

    return train_images_data, train_labels, test_images_data, test_labels

def read_data_sets(data_path,
                   dtype=np.float32,
                   reshape=False,
                   validation_size=1000,
                   seed=None):

    train_images = np.load(os.path.join(data_path, 'train_images_data.npy'))
    train_labels = np.load(os.path.join(data_path, 'train_labels.npy')) # here the labels should contain [digital, digital_color, coor]

    test_images = np.load(os.path.join(data_path, 'test_images_data.npy'))
    test_labels = np.load(os.path.join(data_path, 'test_labels.npy'))

    # print('train_images: ', train_images)
    # print('train_labels: ', train_labels)
    # print('test_images: ', test_images)
    # print('test_labels: ', test_labels)

    # print('train_images shape: ', train_images.shape)
    # print('train_labels shape: ', train_labels.shape)
    # print('test_images shape: ', test_images.shape)
    # print('test_labels shape: ', test_labels.shape)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                         .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    ## the following sentences is used to extract one image, so that we can overfit the model
    # train_images = train_images[0]
    # train_labels = train_labels[0]
    # train_images = np.expand_dims(train_images, 0)
    # train_labels = np.expand_dims(train_labels, 0)


    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return Datasets(train=train, validation=validation, test=test)




if __name__=="__main__":
    #prepare_data(data_path=FLAGS.Data_Dir, train_size=90000)
    prepare_data(data_path=FLAGS.Data_Dir, train_size=16000, num_channels=FLAGS.NUM_CHANNELS, is_save=True)

