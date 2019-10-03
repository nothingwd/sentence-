#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-27 18:28
# @Author  : Sijia Chen (sjia.chen@mail.utoronto.ca)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0

''' python inherent libs '''
import os


''' third parts libs '''
import numpy as np
from gym import spaces
import tensorflow as tf
from gym import spaces
from gym.envs.registration import EnvSpec
from bases.multi_discrete import MultiDiscrete

''' custom libs '''
from bases.box_utils import boxes_transformation

from bases.AnchorBox_solver import AnchorBoxSolver
anchor_box_solver = AnchorBoxSolver()

# multi-agent world
class World(object):
    def __init__(self):

        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0 # this should bt the number of agent, control which agent to communicate
        # position dimensionality
        self.dim_p = 4
        # color dimensionality
        self.dim_color = 3

        # obdervation dim of each agent, this is the combination of all other agent
        self.obs_dim = 128
        # 
        self.is_discrete_action_space = True
        self.is_discrete_communication_space = True

        self.is_pure_world = True # there is not any operations in this world yet.

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def define_world_agent_space(self):
        # configure spaces
        action_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.is_discrete_action_space:
                u_action_space = spaces.Discrete(self.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=[agent.s_range_bottom, agent.r_range_bottom], 
                                            high=[agent.s_range_upper, agent.r_range_upper], shape=(self.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.is_discrete_communication_space:
                c_action_space = spaces.Discrete(self.dim_c)
            else:
                c_action_space = spaces.Box(low=0, high=1, shape=(self.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            else:
                total_action_space.append(spaces.Box(low=0, high=0, shape=(self.dim_c,)))

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                action_space.append(act_space)
            else:
                action_space.append(total_action_space[0])

        return action_space

    def define_world_agent_obs(self, env_data):
        # observation space
        observation_space = []
        for agent in self.agents:
            obs_dim = self.obs_dim
            observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        return observation_space

    def init_agents(self):
        def state_gent():
            ymin = np.random.uniform(low=0, high=1, size=1)
            xmin = np.random.uniform(low=0, high=1, size=1)
            ymax = np.random.uniform(low=ymin, high=1, size=1)
            xmax = np.random.uniform(low=xmin, high=1, size=1)

            return np.array([ymin, xmin, ymax, xmax])

        for agent in self.agents:
            agent.state.spatial_state = state_gent()
            if not agent.silent:
                # communication action
                agent.state.utterance_state = np.ones(self.dim_c) # notice all other agents

            else:
                agent.state.utterance_state = np.zeros(self.dim_c) # keep slience to all other agents

    # update state of the world
    def set_scripted_agents(self, env_target):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, env_target)

    # update state of the world
    def set_policy_agents(self, action_n):
        # set actions for scripted agents
        for i, agent in enumerate(self.policy_agents):
            agent.action.spa_trans_coeff = action_n[0]
            agent.action.utterance = action_n[1]

    def step_world(self):
        ''' operate the action for agent in the world 

            two actions:
                1. transfrom each agent according to the spa_mov action 
                2. send info to agent with corresponding communication action
        '''
        # 1. change the state for each anchor by matrix transform
        for agent in self.agents:
            new_spatial_state = boxes_transformation(agent.state.spatial_state, agent.action.spa_trans_coeff)
            new_utterance_state = agent.action.utterance_state # this is the predicted utterance

            # update the state
            agent.state.spatial_state = new_spatial_state
            if not agent.silent:
                agent.state.utterance_state = new_utterance_state
            else:
                agent.state.utterance_state = np.zeros(self.dim_c)



