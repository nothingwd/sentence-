#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-28 21:13
# @Author  : Sijia Chen (sjia.chen@mail.utoronto.ca)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0

''' python inherent libs '''
import os


''' third parts libs '''
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import VGG16

''' custom libs '''
from bases.box_utils import convert_order, normalize_coors, reverse_normed_coors, boxes_transformation

class ObservationExtractor(object):
    ''' Obtaining the observations from the current env_data '''


    def __init__(self, ROI_F_Fsize=[7, 7], base_extractor_name=None, base_opt_layer_name='conv5_3'):


        self._Fsize_h = ROI_F_Fsize[0]
        self._Fsize_w = ROI_F_Fsize[1]


        self._image_model_transfer = None
        if base_extractor_name is not None:
            if base_extractor_name is 'VGG16':
                image_model = VGG16(include_top=True, weights='imagenet')
                transfer_layer = image_model.get_layer(base_opt_layer_name)
                self._image_model_transfer = Model(inputs=image_model.input,
                                            outputs=transfer_layer.output)

    def set_extract_base_map(self, env_data):
        if self._image_model_transfer is None: # using the input map directly
            self._base_map = env_data

        else:
            self._base_map = self._image_model_transfer.predict(env_data)

        map_size = tf.shape(self._base_map)
        map_h = tf.gather(map_size, [0])
        map_w = tf.gather(map_size, [1])

        env_size = tf.shape(env_data)
        env_h = tf.gather(env_size, 0)
        env_w = tf.gather(env_size, 1)
        # get the shrink scale
        self._extractor_h_scale = tf.cast(env_h / map_h)
        self._extractor_w_scale = tf.cast(env_w / map_w)

        return self._extractor_h_scale, self._extractor_w_scale

    def set_extractor(self, module_name='ROI_fixed_f_layer', op_type='roi_pooling'):
        self._extractor = ROI_fixed_oper_build(module_name, op_type)

    def extract_obs(self, agents_spatial_state):
        ''' extract the observation for this agent from the env_data 

            Args:
                agents_spatial_state: tensor wit shape [num_agents, 4]

            Output:
                obervations: tensor with shape [num_agents, self._Fsize_h, self._Fsize_w, channels]
        '''
        

        obervations = self._extractor(self._base_map, agents_spatial_state)

        return obervations

    def ROI_fixed_oper_build(self, layer_name="ROI_fixed_f_layer", op_type="roi_pooling"):
        def ROI_pooliong(embedded_image_fp, agents_spatial):
            """
            Used to map all the generated anchors into fixed size of feature
            Note: instead of using crop_and_resize function to crop the correspodning region in the embedded_image_fp
                for specific ROI. Since we generate the anchors with same size, so the mapped regions are all the same,
                we use convolution layer with kernel size same as the mapped region to generate
                features for each anchor box.

            Args:
                embedded_image_fp: [1, h, w, C], the global features of the image --> the output feature map
                                                of the function main_model._img_feature_extractor(...).

                mapped_anchors_coors (ROI): the normed coors for each anchors` state
            Output:
                rois_fixed_fd: Tensor with shape, [num_anchors, fixed_h, fixed_w, C]

            """
            num_agents = tf.shape(agents_spatial)[0]
            crops = tf.image.crop_and_resize(image=embedded_image_fp, boxes=agents_spatial, 
                                    box_ind=tf.zeros((num_agents, ), dtype=tf.int32),
                                    crop_size=[self._Fsize_h*2, self._Fsize_w*2],
                                    method='bilinear')
            
            # the output size of this operation is [num_anchors, fsize_h, fsize_w, num_channels]
            rois_fixed_fd = tf.nn.max_pool(crops, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


            return rois_fixed_fd

            # return mapped_anchors_ROI_f

        if op_type == "roi_pooling":
            return ROI_pooliong
        else:
            return None


    def to_tf_standard_ipt(self, anchors_feature_fixed):
        # Converting the shape of the input tensor to the standard type
        # used in Tensorflow [batch_size, h, w, C]
        # Since the batch size in my model is fixed one, so...
        # Convert [fp_h * fp_w, C * fixed_h * fixed_w] to [1, fp_h * fp_w,
        #          1, C * fixed_h * fixed_w]

        anchors_feature_fixed = tf.expand_dims(anchors_feature_fixed, axis=0)
        anchors_feature_fixed = tf.expand_dims(anchors_feature_fixed, axis=2)
        return anchors_feature_fixed
