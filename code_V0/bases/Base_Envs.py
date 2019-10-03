#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-27 17:03
# @Author  : Sijia Chen (sjia.chen@mail.utoronto.ca)
# @Group   : U of T (iQua research group)
# @Github  : https://github.com/CSJDeveloper
# @Version : 0.0

''' python inherent libs '''
import os
import inspect
import logging
import hashlib


''' third parts libs '''
import numpy as np
''' custom libs '''

logger = logging.getLogger(__name__)

from bases.rendering import AnchorBoxVisualizer
from bases.box_utils import reverse_normed_coors

# in this environment, we regard the state of agent and the observatoin the different thing,
#   - state of agent is only the state of this agent but the observation are the information 
#   comes from all other items.

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(object):

    def __init__(self, world, obs_extractor=None,
                 reward_extractor=None,
                 shared_viewer=True):

        self.world = world # the actual world, contains all information needed
        self.world.init_agents()
        # because that the env only useful for the policy-based agents,
        # therefore, the default agents for env are the policy-based agents
        self.agents = self.world.policy_agents # the controlled agents in this world

        # set required vectorized gym env property
        self.n = len(world.policy_agents)

        self.obs_extractor = obs_extractor
        self.reward_extractor = reward_extractor

        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = world.define_world_agent_space() # contain self.n items each of which contain the action space of that agent 
        self.observation_space = world.define_world_agent_obs() # self.n items, each of which is the observation of this agent,

        self.actions_his = [list() for i in range(self.n)] # contain history actions for agent in this environment

        #agent.action.c = np.zeros(self.world.dim_c)
        # agent.action.u = np.zeros(self.world.dim_p)
        # agent.action.c = np.zeros(self.world.dim_c)
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def set_env(self, env_data, env_target, env_target_labels):
        self.env_data = env_data # the actually data for this world (it`s the image in our method)
        self.env_tg = env_target # the target in this environment
        self.env_tg_lbs = env_target_labels

        env_size = tf.shape(self.env_data)
        self.env_h = tf.gather(env_size, [0])
        self.env_w = tf.gather(env_size, [1])
        # inialize the obsercation extracot
        self.obs_extractor.set_extract_base_map(env_data)
        self.reward_extractor.set_reward_base(self.env_tg, self.env_tg_lbs)


    def step(self, action_n):
        """ operate current action in the environment 
            
            Args:
                action_n: a list with length n, each contains two actions for corresponding agent,
                        the action here is actually for the policy-based agent, 
                        for scripy-based agent, only updated in the world object
        """

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
            self.actions_his[i].append(action_n[i])

        # advance world state
        self.world.set_scripted_agents(self.env_tg)
        self.world.set_policy_agents(action_n)

        self.world.step_world() # operate the action in the world for each agent

        # record observation for each agent
        # we only need to get the rewward for policy agents for further learning
        obs_n = self._get_obs(self.agents)
        rewards_n = self._get_reward(self.agents)

        # all agents get total reward in cooperative case
        reward = np.sum(rewards_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        obs_n = self._get_obs(self.agents)

        return obs_n

    # get observation for a particular agent
    def _get_obs(self, agents):
        if self.obs_extractor is None:
            return None
        agents_spatial_state = [agent.state.spatial_state for agent in agents]
        return self.obs_extractor.extract_obs(agents_spatial_state)

    # get reward for a particular agent
    def _get_reward(self, agents):
        if self.reward_extractor is None:
            return 0.0
        agents_spatial_state = [agent.state.spatial_state for agent in agents]
        unnormed_coors = [reverse_normed_coors(sp_t, self.env_h, self.env_w) 
                            for sp_t in agents_spatial_state]
        return self.reward_extractor.IOU_reward(unnormed_coors)

    def _set_action(self, action, agent, time=None):
        ''' operate action for the agent in this env
            action: contains two item, 
                    - first one is the position action
                    - the second one is the communication action

        '''
        if agent.movable:
            # physical action
            agent.action.pos_mov = action[0]

        if not agent.silent:
            # communication action
            agent.action.utterance = action[1]


    # reset rendering assets
    def _reset_render(self):
        self.visualizer = AnchorBoxVisualizer(FLAGS.visual_save_dir)

    # render environment -- plot the environment in the screen
    def render(self):
        agents = self.world.agents
        self.visualizer.visual_iteration_env(env_data=self.env_data, 
                                             env_target=self.env_target, 
                                             agents=agents, 
                                             iter_n=self.time)



# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(object):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
