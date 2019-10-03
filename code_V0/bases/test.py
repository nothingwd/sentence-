#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-28 17:31
# @Author  : Sijia Chen (sjia.chen@mail.utoronto.ca)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0

''' python inherent libs '''
import os


''' third parts libs '''
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from gym import spaces
import numpy as np

''' custom libs '''
tf.enable_eager_execution()


uni = tfd.Uniform(low=[0.5, -0.2], high=[1.5, 0.2])


print(uni)
print(uni.sample((2)))

uni = tfd.Uniform(low=0, high=9)


print(uni)
print(uni.sample((2)))


u_action_space = spaces.Box(low=0, high=0, shape=(6,), dtype=np.float32)
print(u_action_space)
print(u_action_space.sample())