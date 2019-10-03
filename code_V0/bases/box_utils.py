#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-28 20:40
# @Author  : Sijia Chen (sjia.chen@mail.utoronto.ca)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0


''' python inherent libs '''
import os


''' third parts libs '''
import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np

def safe_divide(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)

def normalize_coors(coors, h, w):
    coors = tf.cast(coors, tf.float32)
    scales = [h, w, h, w]
    normed_coors = tf.div(coors, scales)

    return normed_coors

def reverse_normed_coors(normed_coors, h, w):
    normed_coors = tf.cast(normed_coors, tf.float32)
    scales = [h, w, h, w]
    coors = tf.multiply(normed_coors, scales)

    return coors

''' custom libs '''
def convert_order(bboxes):
    ''' Converting the order of bboxes [0, 1, 2, 3] to [1, 0, 3, 2] '''
    boxes = tf.cast(bboxes, tf.float32)
    num_boxes = tf.gather(tf.shape(boxes), 0)
    left_top_y = tf.slice(boxes, [0, 0], [num_boxes, 1])
    left_top_x = tf.slice(boxes, [0, 1], [num_boxes, 1])
    right_bot_y = tf.slice(boxes, [0, 2], [num_boxes, 1])
    right_bot_x = tf.slice(boxes, [0, 3], [num_boxes, 1])

    reordered_bboxes = tf.concat([left_top_x, left_top_y, right_bot_x, right_bot_y], axis=1)
    return reordered_bboxes

def get_boxes_info(boxes):
    boxes = tf.cast(boxes, tf.float32)
    num_boxes = tf.gather(tf.shape(boxes), 0)
    left_top_y = tf.slice(boxes, [0, 0], [num_boxes, 1])
    left_top_x = tf.slice(boxes, [0, 1], [num_boxes, 1])
    right_bot_y = tf.slice(boxes, [0, 2], [num_boxes, 1])
    right_bot_x = tf.slice(boxes, [0, 3], [num_boxes, 1])

    hs = right_bot_y - left_top_y
    ws = right_bot_x - left_top_x

    centers = tf.concat([left_top_y + hs/2, left_top_x + ws/2], axis=1)
    return left_top_y, left_top_x, right_bot_y, right_bot_x, hs, ws, centers

def bboxs_IOU(bboxs_refs, bboxes, name='IOU'):

    try:
        bboxs_refs = tf.convert_to_tensor(bboxs_refs, dtype=tf.float32)
        bboxes = tf.convert_to_tensor(bboxes, dtype=tf.float32)
    except ValueError:
        bboxs_refs = tf.cast(bboxs_refs, tf.float32)
        bboxes = tf.cast(bboxes, tf.float32)

    def ssingle_bboxs_IOU(bboxs_ref):
        """Compute the IOU between a reference box and a collection of bounding boxes.
        Args:
          bboxs_ref: (1, 4) or (N, 4) Tensor with reference bounding box.
                    [xmin, ymin, xmax, ymax]
          bboxes: (N, 4) Tensor, collection of bounding boxes.
                    [xmin, ymin, xmax, ymax]
        Return:
          (1, N) Tensor with IOU score.
        """

        # 1. the map_fn function is called, so unpack of bboxes_ref on dimension 0
        #   is (4). Therefore we should change its shape to 91, 4
        bboxs_ref = tf.reshape(bboxs_ref, (1, 4))
        bboxs_ref_num = tf.gather(tf.shape(bboxs_ref), 0)
        bbox_num = tf.gather(tf.shape(bboxes), 0)

        ref_upper_left_x = tf.slice(bboxs_ref, begin=[0, 0], size=[bboxs_ref_num, 1])
        ref_upper_left_y = tf.slice(bboxs_ref, begin=[0, 1], size=[bboxs_ref_num, 1])
        ref_bottom_right_x = tf.slice(bboxs_ref, begin=[0, 2], size=[bboxs_ref_num, 1])
        ref_bottom_right_y = tf.slice(bboxs_ref, begin=[0, 3], size=[bboxs_ref_num, 1])
        ref_h = ref_bottom_right_y - ref_upper_left_y
        ref_w = ref_bottom_right_x - ref_upper_left_x

        upper_left_x = tf.slice(bboxes, begin=[0, 0], size=[bbox_num, 1])
        upper_left_y = tf.slice(bboxes, begin=[0, 1], size=[bbox_num, 1])
        bottom_right_x = tf.slice(bboxes, begin=[0, 2], size=[bbox_num, 1])
        bottom_right_y = tf.slice(bboxes, begin=[0, 3], size=[bbox_num, 1])
        h = bottom_right_y - upper_left_y
        w = bottom_right_x - upper_left_x

        with tf.name_scope(name + "_IOU"):
            # 1. Getting the intersection rectangle of the corresponding two bboxs
            int_left_upper_x = tf.maximum(upper_left_x, ref_upper_left_x)
            int_left_upper_y = tf.maximum(upper_left_y, ref_upper_left_y)

            int_bottom_right_x = tf.minimum(bottom_right_x, ref_bottom_right_x)
            int_bottom_right_y = tf.minimum(bottom_right_y, ref_bottom_right_y)
            inter_h = int_bottom_right_y - int_left_upper_y
            inter_w = int_bottom_right_x - int_left_upper_x
            inter_square = tf.multiply(inter_h, inter_w)
            inter_square = tf.where(tf.greater(inter_square, 0), inter_square, tf.zeros_like(inter_square))
            inter_square = tf.where(tf.greater(inter_h, 0), inter_square, tf.zeros_like(inter_square))
            inter_square = tf.where(tf.greater(inter_w, 0), inter_square, tf.zeros_like(inter_square))

            union_square = tf.multiply(h, w) + tf.multiply(ref_h, ref_w) - inter_square
            inter_square = tf.cast(inter_square, tf.float32)
            union_square = tf.cast(union_square, tf.float32)
            calcIOU = tfe_math.safe_divide(inter_square, union_square, "calc_IOU")
            # 2. making the no overlap bbox to zero
            calcIOU = tf.where(tf.greater(calcIOU, 0), calcIOU, tf.zeros_like(calcIOU))

        calcIOU = tf.transpose(calcIOU)

        return calcIOU


    calcIOUs = tf.map_fn(ssingle_bboxs_IOU, bboxs_refs)
    calcIOUs = tf.squeeze(calcIOUs, 1)
    return calcIOUs

def boxes_transformation( normed_boxes, predicted_tras_coefficients):
    ''' transfrom the boxes acording to the predicted coefficients 

        Args:
            normed_boxes: normalized boxes with shape [num_boxes, 4] [y_min, x_min, y_max, x_max] 
                        it is the spatial_state for current agent
            predicted_tras_coefficients: predicted transform coefficients [num_boxes, 4] 4: 
                                        [s1, s2, r1, r2]
    '''


    normed_boxes = tf.cast(normed_boxes, tf.float32)
    num_boxes = tf.gather(tf.shape(normed_boxes), 0)
    boxes_num_range = tf.range(num_boxes)
    # construct the transform matrix
    def construct_transform_matrix(transform_coefficient):
        # input is a tensor with shape [4] --> [s1, s2, r1, r2]
        transform_coefficient = tf.reshape(transform_coefficient, (-1,))
        base_r_mt = tf.eye(3)
        coe_r = tf.slice(transform_coefficient, [2], [2])
        coe_r = tf.reshape(coe_r, (-1, 1))
        paddings = tf.constant([[0, 1], [2, 0]])
        oper_coe_r = tf.pad(coe_r, paddings, "CONSTANT")
        r_mt = base_r_mt + oper_coe_r

        base_s_mt = tf.matrix_diag(tf.constant([0, 0, 1], tf.float32))
        coe_s = tf.slice(transform_coefficient, [0], [2])
        diag_coe_s = tf.matrix_diag(coe_s)
        paddings = tf.constant([[0, 1], [0, 1]])
        oper_diag_coe_s = tf.pad(diag_coe_s, paddings, "CONSTANT")
        s_mt = base_s_mt + oper_diag_coe_s

        transform_matrix = tf.matmul(r_mt, s_mt)

        return transform_matrix

    def box_transform(box_idx):
        box_idx = tf.cast(box_idx, tf.int32)
        # get coordinate of the box [ymin, xmin, ymax, xmax]
        box_coor = tf.slice(normed_boxes, [box_idx, 0], [1, 4])
        box_trans_coe = tf.slice(predicted_tras_coefficients, [box_idx, 0], [1, 4])

        top_left_pos = tf.slice(box_coor, [0, 0], [1, 2])
        top_left_pos = tf.reshape(top_left_pos, [-1, 1])
        top_left_pos_h = tf.pad(top_left_pos, [[0, 1], [0, 0]], "CONSTANT", constant_values=1)

        bottom_right_pos = tf.slice(box_coor, [0, 2], [1, 2])
        bottom_right_pos = tf.reshape(bottom_right_pos, [-1, 1])
        bottom_right_pos_h = tf.pad(bottom_right_pos, [[0, 1], [0, 0]], "CONSTANT", constant_values=1)

        transform_matrix = construct_transform_matrix(box_trans_coe)

        transformed_top_left_pos = tf.matmul(transform_matrix, top_left_pos_h)
        transformed_top_left_pos = tf.slice(transformed_top_left_pos, [0, 0], [2, 1])
        transformed_top_left_pos = tf.reshape(transformed_top_left_pos, [1, 2])

        transformed_bottom_right_pos = tf.matmul(transform_matrix, bottom_right_pos_h)
        transformed_bottom_right_pos = tf.slice(transformed_bottom_right_pos, [0, 0], [2, 1])
        transformed_bottom_right_pos = tf.reshape(transformed_bottom_right_pos, [1, 2])

        transformed_box = tf.concat([transformed_top_left_pos, transformed_bottom_right_pos], axis=1)
        transformed_box = tf.reshape(transformed_box, [-1, ])
        return transformed_box

    normed_transformed_boxes = tf.map_fn(box_transform, tf.cast(boxes_num_range, tf.float32))
    print('normed_transformed_boxes: ', normed_transformed_boxes)
    
    # print(ok)
    return normed_transformed_boxes

if __name__=="__main__":
    tf.enable_eager_execution()

    bboxs11 = tf.constant([[17,  7, 37, 30.],
                         [16, 10, 41, 30.],
                         [75, 37, 94, 62.],
                         [ 9, 59, 32, 80.],
                         [11, 60, 28, 77.],
                         [64, 61, 89, 84.],
                         [63, 60, 84, 83.],
                         [62, 60, 84, 81.]])
    bboxs1 = tf.constant([[15,  9, 37, 31.],
                         [63, 61, 86, 84.],
                         [10, 59, 29, 78.],
                         [76, 37, 98, 59.]])


    ious = bboxs_IOU(bboxs_refs=bboxs1, bboxes=bboxs11)


    print(ious)
