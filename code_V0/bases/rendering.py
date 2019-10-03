#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-27 18:47
# @Author  : Sijia Chen (sjia.chen@mail.utoronto.ca)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0

''' python inherent libs '''
import os
from collections import OrderedDict

''' third parts libs '''
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import scipy.misc

''' custom libs '''
from bases.box_utils import convert_order, normalize_coors, reverse_normed_coors, bboxs_IOU


COLOR = OrderedDict([("red", "#FF0000"),
                     ("blue", "#0000FF"),
                     ("green", "#008000"),
                     ("purple", "#800080"),
                     ("yellow", "#FFFF00"),
                     ('saddlebrown', '#8B4513'),
                     ('brown', '#A52A2A'),
                     ("aqua", "#00FFFF"),
                     ('midnightblue', '#191970'),
                     ('darkgreen', '#006400'),
                     ('darkblue', '#00008B')])


colors_digial_map = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'purple': 4, 'saddlebrown': 5, 'brown': 6, 'aqua': 7}
digital_colors_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple', 5: 'saddlebrown', 6: 'brown', 7: 'aqua'}


class AnchorBoxVisualizer(object):
    ''' Visualing the generated anchors and boxs '''

    def __init__(self, to_save_visual_dir):
        self._to_save_visual_dir = to_save_visual_dir

    def _draw_single_box(self, image_b, coordinate, **sub_kwargs):
        drawObject = ImageDraw.Draw(image_b)

        if "outline" in sub_kwargs:
            drawObject.rectangle(coordinate, outline=sub_kwargs['outline'])
        elif "fill" in sub_kwargs:
            drawObject.rectangle(coordinate, fill=sub_kwargs['fill'])
        else:
            drawObject.rectangle(coordinate, fill='#FF0000')

        if "text" in sub_kwargs:
            text = sub_kwargs['text']
            try:
                text_color = sub_kwargs['text_color']
            except:
                text_color = '#FF0000'
            Font = ImageFont.truetype(os.path.join(os.getcwd(), 'visualization/font/timesbd.ttf'), 10, index=0)
            text_start_posx = coordinate[0]
            text_start_posy = coordinate[1] + 3
            text_w, text_h = Font.getsize(text)
            text_upper_left = (text_start_posx, text_start_posy)
            text_upper_right = (text_start_posx + text_w, text_start_posy + text_h)
            drawObject.rectangle(text_upper_left + text_upper_right, fill="#FFFFFF")
            drawObject.text(text_upper_left, text, fill=text_color, font=Font)
            # drawObject.rectangle(upper_left + bottom_right, outline=(255, 0, 0))
        return image_b

    def _draw_as_blend(self, img0, image_board, box_coordinate, color, **kwargs):
        img0 = self._draw_single_box(img0, box_coordinate, outline=color, **kwargs)
        image_board = self._draw_single_box(image_board, box_coordinate, fill=color, **kwargs)
        
        cur_img1 = img0.convert('RGBA')
        cur_img2 = image_board.convert('RGBA')

        cur_img_final = Image.blend(cur_img1, cur_img2, 0.3)
        cur_img_final = cur_img_final.convert('RGB')
        return cur_img_final, img0, image_board

    def draw_image_boxs(self, image_board, box_coordinates, color_id, **kwargs):
        '''
        Draw the box  in the image
        Args:
            image_board: Image data, the image needed to be drawed. with shape [width, height]
            box_coordinates: list, each element is a list which containing the boudning
                                box coordinations([xmin, ymin, xmax, ymax]) of the input phrase
                                --> xy – Two points to define the bounding box. Sequence of either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]
            color_id: int, the color of the bounding box
        '''


        color = digital_colors_map[color_id[0]]
        color = COLOR[color]

        img0 = image_board.copy()
        for box_idx in range(len(box_coordinates)):
            box_coordinate = box_coordinates[box_idx]
            cur_img_final, img0, image_board = self._draw_as_blend(img0, image_board, box_coordinate, color, **kwargs)

            if 'is_save' in kwargs and kwargs['is_save']:
                cur_img_final.save(os.path.join(self._to_save_visual_dir, kwargs['save_pre'] + str(box_idx) + '.jpg'))
            
        img1 = img0.convert('RGBA')
        img2 = image_board.convert('RGBA')

        img_final = Image.blend(img1, img2, 0.4)
        img_final = img_final.convert('RGB')
        return img_final


    def visual_iteration_env(self, env_data, env_target, agents, iter_n, save_pre='visual_test_'):
        ''' visual the environment in current iteration '''

        tg_board = env_data[0] # [h, w, 3]
        target_coors = env_target[0] # [num_phrases, 4]

        h = tf.gather(tf.shape(env_data), [0])
        w = tf.gather(tf.shape(env_data), [1])

        img = scipy.misc.toimage(tg_board)
        img_array = np.array(img)
        img = Image.fromarray(img_array)

        img_board = img.copy()
        img0 = img_board.copy()

        for agent in agents:
            iter_box_norm = agent.state.spatial_state
            box_coordinate = reverse_normed_coors(iter_box_norm, h, w)
            box_coordinate_array = np.reshape(box_coordinate, (1, 4))

            ious = bboxs_IOU(target_coors, box_coordinate_array)
            iou = np.max(ious).numpy()

            cur_img_final, img0, img_board = self._draw_as_blend(img0=img0, image_board=img_board, box_coordinate=box_coordinate, 
                                                                 text='%.2f' % iou)

        img1 = img0.convert('RGBA')
        img2 = img_board.convert('RGBA')

        img_final = Image.blend(img1, img2, 0.4)
        img_final = img_final.convert('RGB')
        img_final.save(os.path.join(self._to_save_visual_dir, save_pre + '_iter_' + str(iter_n) + 'boxes.jpg'))

        