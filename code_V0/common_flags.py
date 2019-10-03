#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-29 04:19
# @Author  : Sijia Chen (sjia.chen@mail.utoronto.ca)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0


''' python inherent libs '''
import sys
import logging
import os
import collections
import re

''' third parts libs '''
from tensorflow.python.platform import flags

import tensorflow as tf


''' local custom libs '''

"""Define flags are common for both train.py and eval.py scripts."""


FLAGS = flags.FLAGS

logging.basicConfig(
    level=logging.ERROR, #DEBUG # INFO
    stream=sys.stderr,
    format='%(levelname)s '
    '%(asctime)s.%(msecs)06d: '
    '%(filename)s: '
    '%(lineno)d '
    '%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


# The child path can be any type
def path_compatible_system(base_path, added_chind_path):
    pattern_win = re.compile('/')
    pattern_lx = re.compile('\\\\')

    win_child_path = re.sub(pattern_win, '\\\\', added_chind_path)
    lx_child_path = re.sub(pattern_lx, '/', added_chind_path)

    if "\\" in base_path and not "/" in base_path:
        # Windows path
        compat_path = os.path.join(base_path, win_child_path)

    if "/" in base_path and not "\\" in base_path:
        # linux path
        compat_path = os.path.join(base_path, lx_child_path)

    return compat_path

def get_common_path():
    # get the current path
    cur_path = os.getcwd()
    # parent's path for current path
    cur_project_path = os.path.abspath(os.path.dirname(cur_path)
                                +os.path.sep+".")
    # The previous two levels of the current file
    main_dir = os.path.abspath(os.path.dirname(cur_project_path)+os.path.sep+"..")

    sources_path = os.path.join(main_dir, "sources")

    return main_dir, sources_path, cur_project_path

MAIN_dir_p, SOURCES_p, CUR_PROJECT_p = get_common_path()

first_idx = SOURCES_p.find('ResearchArea')
SOURCES_p = SOURCES_p[: first_idx+len('ResearchArea')]
SOURCES_p = os.path.join(SOURCES_p, 'sources')



def define():
    """Define common flags."""

    # yapf: disable
    ###################
    # DataSets setting #
    ###################
    # --> We do not allow for setting the dataset by yourself!
    #       you must design a config for your dataset
    # flags.DEFINE_string('dataset_dir', None,
    #                     'Dataset root folder.')
    flags.DEFINE_string('dataset_name', 'flickr30k_entities',
                        'Name of the dataset. Supported: coco, flickr30k_entities')
    flags.DEFINE_string('split_name', 'train',
                        'Dataset split name to run evaluation for: test,train.')

    #########################
    # Main model Parameters #
    #########################
    flags.DEFINE_integer('tensor_common_space_size', 64,
                         'The size of predefined common space in tensor module')
    flags.DEFINE_integer('final_common_space_size', None,
                         'The size of the final Learned common space')
    flags.DEFINE_string('anchor_size', "128,128",
                        'size of anchors in the main model')
    flags.DEFINE_string('ROI_Fixed_Fsize', "5,5",
                        'size of features of each ROI ')
    flags.DEFINE_integer('global_itera_num', 5,
                         'the number of the iteration used ')
    flags.DEFINE_string('Anchor_fd_layers', '4096, 128,',
                        'The layers used to process anchors to specific space size')
    flags.DEFINE_string('Phrase_fd_layers', '128',
                        'The layers used to process txt to specific space size')

    ###########################
    # Image Fmodel Parameters #
    ###########################
    flags.DEFINE_string('pretrained_IFM_dir',
                        path_compatible_system(SOURCES_p,
                                            'prepared_models/tf_pretrained_nets'),
                        'Directory where to load the pretrained model.\
                    Note: This is part of master model that used as a feature extractor')

    flags.DEFINE_string('pretrained_IFM_name',
                        'vgg_16',
                        'Name of the pretained module.\
                        Note: This is part of master model that used as a feature extractor')
    flags.DEFINE_string('final_endpoint', 'conv5_3', # pool5
                        'Endpoint to cut inception tower')

    flags.DEFINE_bool('use_augment_input', True,
                      'If True will use image augmentation')

    ##########################
    # Text Fmodel Parameters #
    ##########################
    flags.DEFINE_string('pretrained_TE_file',
                        path_compatible_system(SOURCES_p, "Embedding_Vectors/glove_6B/glove_6B_"),
                        'Name of the pretained module.\
                        Note: This is part of master model that used as a feature extractor')
    flags.DEFINE_integer("embedding_dim", 300,
                         "Dimensionality of character embedding (default: 300)")
    flags.DEFINE_integer("max_sequence_length", 57,
                         "the max length of the sequence...")
    flags.DEFINE_integer("max_phrases_number", 5,
                         "the max number of the phrases...")
    flags.DEFINE_integer("vocab_size", 20215,  # should be equal to the dataset used
                         "the number of words in the vocabulary")
    flags.DEFINE_integer("encode_wordSyntax", 0,
                         "whether to combine word syntax with CNN")

    flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

    ####################
    # Train Parameters #
    ####################
    flags.DEFINE_integer("number_of_steps", 2000000,
                     "Number of training iterations (default: 200w)")
    flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    flags.DEFINE_integer('max_number_of_steps', int(1e10),
                     'The maximum number of gradient steps.')
    # flags.DEFINE_integer('num_samples_train', 10000,
    #                      "Total number of samples for training")
    flags.DEFINE_integer('batch_size', 1,
                         'Batch size. (Only support 1 currently!)')
    flags.DEFINE_integer('NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN', 400000,
                         'Total number of examples in training')
    flags.DEFINE_float('initial_learning_rate', 0.04,
                       'initial learning rate')
    flags.DEFINE_float('learning_rate_decay_factor', 0.99,
                       'learning rate decay factor')
    flags.DEFINE_float('num_epochs_per_decay', 2,
                       'per decay within num rpochs')
    flags.DEFINE_string('optimizer', 'momentum',
                        'the optimizer to use, should be existed in\
                        [Adam, RMSProp, Ftrl, Adagrad, Momentum, SGD]')
    flags.DEFINE_float('momentum', 0.9,
                        'momentum value for the momentum optimizer if used')
    flags.DEFINE_float('clip_gradient_norm', 2.0,
                       'If greater than 0 then the gradients would be clipped by '
                       'it.')

    flags.DEFINE_string('train_log_dir', './logs/train_logs',
                        'Directory where to write event logs.')
    flags.DEFINE_integer("evaluate_every", 100,
                         "Evaluate model on dev set after this many steps (default: 100)")
    flags.DEFINE_integer("iter_log_frequency", 100,
                         "Log the information of training model after this many steps (default: 100)")
    flags.DEFINE_integer("max_checkpoints_to_keep", 5000,
                         "Save model after this many steps (default: 100)")
    flags.DEFINE_integer('save_summaries_secs', 60,
                         'The frequency with which summaries are saved, in '
                         'seconds.')
    flags.DEFINE_integer('save_interval_secs', 600,
                         'Frequency in seconds of saving the model.')

    flags.DEFINE_string('checkpoint_inception', '',
                        'Checkpoint to recover inception weights from.')

    flags.DEFINE_string('master',
                      '',
                      'BNS name of the TensorFlow master to use.')
    flags.DEFINE_integer('task', 0,
                         'The Task ID. This value is used when training with '
                         'multiple workers to identify each worker.')
    flags.DEFINE_integer('ps_tasks', 0,
                         'The number of parameter servers. If the value is 0, then'
                         ' the parameters are handled locally by the worker.')
    flags.DEFINE_bool('sync_replicas', False,
                      'If True will synchronize replicas during training.')
    flags.DEFINE_integer('replicas_to_aggregate', 1,
                         'The number of gradients updates before updating params.')
    flags.DEFINE_integer('total_num_replicas', 1,
                         'Total number of worker replicas.')
    flags.DEFINE_integer('startup_delay_steps', 15,
                         'Number of training steps between replicas startup.')
    flags.DEFINE_boolean('show_graph_stats', False,
                         'Output model size stats to stderr.')


    # 'sequence_loss_fn'
    flags.DEFINE_float('label_smoothing', 0.1,
                       'weight for label smoothing')

    # yapf: enable




