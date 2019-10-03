#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-29 03:38
# @Author  : Sijia Chen (sjia.chen@mail.utoronto.ca)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0

''' python inherent libs '''
import os


''' third parts libs '''
import tensorflow as tf

''' custom libs '''
from bases.box_utils import bboxs_IOU

class RewardExtractor(object):
    def __init__(self):
        pass


    def set_reward_base(self,  env_target, env_target_labels, h_scale, w_scale):
        self.env_target = env_target # tensor with size [num_phrases, ]
        self.h_scale = h_scale
        self.w_scale = w_scale
        self.env_target_labels = env_target_labels

        self.process_bboxs_label(env_target_labels)

    def process_bboxs_label(self, bboxs_label):
        bboxs_label = tf.cast(bboxs_label, tf.float32)
        phrases_bboxs_label_flatten = tf.layers.flatten(bboxs_label)
        phrases_bboxs_label_flatten = tf.gather(phrases_bboxs_label_flatten, 0)

        unique_pblf, unique_pblf_idx = tf.unique(phrases_bboxs_label_flatten)
        self._unique_bboxs_label = unique_pblf
        self._unique_bboxs_label_idx = unique_pblf_idx
        self._phrases_bboxs_label_flatten = phrases_bboxs_label_flatten

    def IOU_reward(self, anchors_coordinate):
        """
        Adding labels for each anchor according to the correspodning phrase
        The soft lebles are calculated as the IOU between the generated bboxs and the groundtruth bboxs

        Args:
            anchors_coordinate: int32 Tensor with shape [fp_h * fp_w, 4],
                                [upper_y, upper_x, bottom_y, bottom_x]
            bt_gt_bboxs_coordinate: int64 Tensor,  with shape [1, total_num_gt_bboxs, 4],
                                [xmin, ymin, xmax, ymax] Storing all the bboxs in current image
            bboxs_label: Tensor with shape [1, total_num_gt_bboxs, 1]
                                            Storing label for each bboxs which also maintain the
                                            same order as the phrase
        Output:
            grouped_labels_IOU_score: Tensor with shape [num_phrases, num_anchors]
        """
        bt_gt_bboxs_coordinate = self.env_target
        # 0. convert the dtype into consistent
        bt_gt_bboxs_coordinate = tf.cast(bt_gt_bboxs_coordinate, tf.float32)

        # 0.1 convert the shape of tensors into standard
        # That is instead of listing each bbox, we gather bboxs for each label
        # then there are several bboxs for each label
        bt_gt_bboxs_coordinate = tf.gather(bt_gt_bboxs_coordinate, indices=0)  # each batch

        # 0.2 Converting the type of anchors_coordinate to [upper_x, upper_y, bottom_x, bottom_y]
        #       such that we can calculate the IOU score
        reordered_anchors_coordinate = self._reorder_anchors_type(anchors_coordinate)

        # 1. calculating the IOU score between anchors and each gt_bboxes
        # 1.1 The map_fn is used to parallelly operate the calculate the score
        #       the return shape should be a list with length total_num_gt_bboxs and each
        #       element contained should be (1, total_num_anchors)
        #       The total shape of the Tensor should be (total_num_gt_bboxs, 1, total_num_anchors)
        try:
            phrases_IOU_score = tf.map_fn(lambda x: bboxes_utils.bboxs_IOU(x,
                                                                           reordered_anchors_coordinate,
                                                                           "anchors_gb"),
                                          bt_gt_bboxs_coordinate)
        except:
            phrases_IOU_score = tf.map_fn(lambda x: bboxes_utils.bboxs_IOU(x,
                                                                           reordered_anchors_coordinate,
                                                                           "anchors_gb"),
                                          bt_gt_bboxs_coordinate)
        try:                                              
            # 1.2 Converting to (total_num_gt_bboxs, total_num_anchors)
            phrases_IOU_score = tf.squeeze(phrases_IOU_score, axis=1)
        except:
            phrases_IOU_score = tf.squeeze(phrases_IOU_score, axis=1)       

        # 2. Integrate the IOU score for each gt_bboxes according to the corresponding label
        #   namely, calculate the IOU score anchors and each phrase
        def integrate_IOU(uiq_pblf):
            phrase_label_mask = tf.equal(self._phrases_bboxs_label_flatten,
                                         tf.ones_like(self._phrases_bboxs_label_flatten) * uiq_pblf)
            group_phs_IOUs = tf.boolean_mask(phrases_IOU_score, phrase_label_mask)
            group_phs_IOUs = tf.reduce_max(group_phs_IOUs, 0)
            return group_phs_IOUs

        grouped_labels_IOU_score = tf.map_fn(lambda x: integrate_IOU(x), self._unique_bboxs_label)
        
        return grouped_labels_IOU_score