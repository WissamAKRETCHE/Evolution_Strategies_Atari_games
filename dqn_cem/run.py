#!/usr/bin/env python

from __future__ import division
import colored_traceback.always
from PIL import Image
import numpy as np

import tensorflow as tf
from config import parse_flags
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.memory import EpisodeParameterMemory
from rl.core import Agent, Processor
from rl.agents.dqn import DQNAgent
from rl.agents.cem import CEMAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from drl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from drl.memory import SequentialMemory
FLAGS = parse_flags()

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


def make_env():
    env = gym.make(FLAGS.game)
    FLAGS.num_actions = env.action_space.n
    FLAGS.agent = "dqn"
    return env


def make_model():
    if FLAGS.model == "3dense":
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(FLAGS.num_actions))
        model.add(Activation('linear'))
    elif FLAGS.model == "dqn_atari":
        input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=input_shape))
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(FLAGS.num_actions))
        model.add(Activation('linear'))
    elif FLAGS.model == "no_hidden":
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(FLAGS.num_actions))
        model.add(Activation('softmax'))
    else:
        raise BaseException("unknown model, please check your flag [--model]")
    print(model.summary())
    return model


def make_policy():
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)
    return policy


def make_memory():
    if FLAGS.memory == "sequential":
        memory = SequentialMemory(limit=FLAGS.memory_limit, window_length=1)
    else:
        raise BaseException("unknown memory type, please check your flag [--memory]")
    return memory


def make_agent():
    policy = make_policy()
    model = make_model()
    if FLAGS.agent == "dqn":
        memory = SequentialMemory()
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.1, value_min=.1, value_test=.05,
                                      nb_steps=1)
        processor = AtariProcessor()
        agent = DQNAgent(model=model, nb_actions=FLAGS.num_actions, policy=policy, memory=memory,
                       processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                       train_interval=4, delta_clip=1.)
        agent.compile(Adam(lr=.00025), metrics=['mae'])
    elif FLAGS.agent == "cem":
        memory = EpisodeParameterMemory(limit=1000, window_length=1)
        processor = AtariProcessor()
        agent = CEMAgent(model=model, nb_actions=FLAGS.num_actions, memory=memory,processor=processor, batch_size=100, nb_steps_warmup=10,
                         train_interval=50, elite_frac=0.05)
        agent.compile()
    else:
        raise BaseException("unknown agent, please check your flag [--agent]")
    return agent


env = make_env()

agent = make_agent()
if FLAGS.mode == "train":
    weights_filename = FLAGS.exp_dir + 'weights.h5f'
    checkpoint_weights_filename = FLAGS.exp_dir + 'weights_{step}.h5f'
    log_filename = FLAGS.exp_dir + 'log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=FLAGS.save_weight_interval)]
    agent.fit(env, callbacks=callbacks, nb_steps=10000000, log_interval=10000,  verbose=2)
    agent.test(env, nb_episodes=5, visualize=True)
elif FLAGS.mode == "test":
    weights_filename = "examples/" + "test.h5f"
    agent.load_weights(weights_filename)
    agent.test(env, nb_episodes=10, visualize=True)
else:
    raise BaseException("unknown mode, please check your flag [--mode]")
