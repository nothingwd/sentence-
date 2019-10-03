#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-29 02:30
# @Author  : Sijia Chen (sjia.chen@mail.utoronto.ca)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0

''' python inherent libs '''
import os


''' third parts libs '''
import tensorflow as tf
import numpy as np


''' custom libs '''
from bases.box_utils import bboxs_IOU


class AnchorBoxSolver(object):
    ''' This is to generate anchors for the input image
        - generating anchors according to the size of image and predefined hyperparameters

        Inputs:
            anchor_hs, anchor_ws: int for size of based anchor
            anchor_scales: the room scale corresponding based anchor
            anchor_aspect_ratios: the proporation of height and width
            anchor_stride_h: here are two understanding of anchor_stride_h
                - This is the distance between centers of two anchors
                - This is the corresponding size between original image and the board, feature map...
                    that you want to genearte anchors. original_image_size / board_size
                Note: if we can generate anchors for every positions in the board, feature map, then if we map
                        these anchors to the original image or another board. It is obviously that the mapped scale
                        is the distance between centres of anchors!! 
                --> If we want to generate boxesin target_board, the distance between near boxes is self._anchor_stride,
                    we can map the target borad to the anchor board by self._anchor_stride and generate anchors 
                    for each position of this anchor board,
                    then we get boxes we want in the target board
                --> another is easy to understandingg, we want to generate 
    '''
    def __init__(self):
        pass


    def get_anchor_board_size(self, target_board):
        ''' Getting the board used to generate anchors according to the relative stride between the board, feature map or map you want to create anchors 
            and the original image. this is stride is img_size_h / board_size, img_size_w / board_size_w
            target_board: a tensor or list
        '''
        target_board_h = target_board_w = None

        target_board = tf.convert_to_tensor(target_board)

        opt_size = target_board.get_shape().as_list()
        [target_board_h, target_board_w] = [opt_size[1], opt_size[2]]

        anchor_board_h = tf.ceil(target_board_h / self._anchor_stride[0])
        anchor_board_w = tf.ceil(target_board_w / self._anchor_stride[1])

        return [anchor_board_h, anchor_board_w]

    def filter_boxes(self, generated_boxes, target_board, scope_name="bboxs_filter"):
        ''' filter anchors whose coordinate is not exieted in the target board 
            
            
        '''
        with tf.name_scope(scope_name):
            num_boxes = tf.reshape(tf.gather(tf.shape(generated_boxes), [0]), [])
            generated_boxes = generated_boxes

            target_board = tf.convert_to_tensor(target_board)
            opt_size = target_board.get_shape().as_list()
            [target_board_h, target_board_w] = [opt_size[1], opt_size[1]]

            bboxs_ymin = tf.slice(generated_boxes, [0, 0], [num_boxes, 1])
            bboxs_xmin = tf.slice(generated_boxes, [0, 1], [num_boxes, 1])
            bboxs_ymax = tf.slice(generated_boxes, [0, 2], [num_boxes, 1])
            bboxs_xmax = tf.slice(generated_boxes, [0, 3], [num_boxes, 1])

            bboxs_ymin = tf.where(bboxs_ymin>0, bboxs_ymin, tf.zeros_like(bboxs_ymin))
            bboxs_xmin = tf.where(bboxs_xmin>0, bboxs_xmin, tf.zeros_like(bboxs_xmin))
            bboxs_ymax = tf.where(bboxs_ymax>0, bboxs_ymax, tf.zeros_like(bboxs_ymax))
            bboxs_xmax = tf.where(bboxs_xmax>0, bboxs_xmax, tf.zeros_like(bboxs_xmax))

            bboxs_ymin = tf.where(bboxs_ymin>target_board_h, target_board_h*tf.ones_like(bboxs_ymin)-1, bboxs_ymin)
            bboxs_xmin = tf.where(bboxs_xmin>target_board_w, target_board_w*tf.ones_like(bboxs_xmin)-1, bboxs_xmin)
            bboxs_ymax = tf.where(bboxs_ymax>target_board_h, target_board_h*tf.ones_like(bboxs_ymax)-1, bboxs_ymax)
            bboxs_xmax = tf.where(bboxs_xmax>target_board_w, target_board_w*tf.ones_like(bboxs_xmax)-1, bboxs_xmax)

            # filtered_boxes = tf.boolean_mask(generated_boxes, bound_mask)
            filtered_boxes = tf.concat([bboxs_ymin, bboxs_xmin, bboxs_ymax, bboxs_xmax], axis=1)

        return filtered_boxes

    def invaild_boxes_remove(self, generated_boxes):
        ''' remove invalid boxes '''
        num_boxes = tf.reshape(tf.gather(tf.shape(generated_boxes), [0]), [])
        generated_boxes = generated_boxes

        bboxs_ymin = tf.slice(generated_boxes, [0, 0], [num_boxes, 1])
        bboxs_xmin = tf.slice(generated_boxes, [0, 1], [num_boxes, 1])
        bboxs_ymax = tf.slice(generated_boxes, [0, 2], [num_boxes, 1])
        bboxs_xmax = tf.slice(generated_boxes, [0, 3], [num_boxes, 1])        

        y_mask = tf.greater(bboxs_ymax, bboxs_ymin)
        x_mask = tf.greater(bboxs_xmax, bboxs_xmin)
        remove_mask = tf.logical_and(y_mask, x_mask)

        # yy_mask = tf.not_equal(bboxs_ymax, bboxs_xmax)
        # xx_mask = tf.not_equal(bboxs_xmax, bboxs_ymin)

        remove_mask = tf.reshape(remove_mask, [-1,])

        return remove_mask
