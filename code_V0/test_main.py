#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-19 20:23
# @Author  : Sijia Chen (csjdeveloper@gamil.com)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0

''' python inherent libs '''
import sys
import json
import os
from datetime import datetime
import time
import math

''' third parts libs '''
import tensorflow as tf
import numpy as np
import cv2

''' custom libs '''
from bases.Base_Envs import MultiAgentEnv
from bases.Base_world import World
from bases.Observation_extractor import ObservationExtractor
from bases.Reward_extractor import RewardExtractor
from datasets.QMNIST.data_provider import read_data_sets



tf.enable_eager_execution()

FLAGS = tf.app.flags.FLAGS

#dataset_holder = read_data_sets(data_path=FLAGS.Data_Dir)

# batch_imgs, batch_labels = dataset_holder.train.next_batch(FLAGS.batch_size)

world = World()


