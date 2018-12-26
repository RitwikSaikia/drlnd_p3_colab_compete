#!/usr/bin/env python3

import logging
import os

import numpy as np

from env import UnityEnv
from maddpg_agent import MADDPGAgent
from utils import to_tensor

logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().setLevel(logging.DEBUG)


def run(config, env):
    checkpoints_dir = "%s/%s" % (config.checkpoints_dir, env.name)
    model_prefix = "%s/model.solved" % checkpoints_dir

    maddpg = MADDPGAgent(env)
    maddpg.load(model_prefix)

    maddpg.eval_mode()

    for i_episode in range(config.num_episodes):
        episode_rewards = np.zeros(maddpg.num_agents)
        states = env.reset(train_mode=False)
        env.render()
        for i_step in range(config.max_steps):
            states = [to_tensor(states[i]) for i in range(maddpg.num_agents)]
            actions = maddpg.step(states, add_noise=False)
            actions = [ac.data.numpy().flatten() for ac in actions]
            states, rewards, dones, infos = env.step(actions)
            episode_rewards += rewards
            if np.any(dones):
                break
            env.render()

        score = np.max(episode_rewards)
        logger.info("Episode %i/%i: Score: %.3f, Steps: %d" % (i_episode + 1, config.num_episodes, score, i_step + 1))

    env.close()


if __name__ == '__main__':
    class Config:
        checkpoints_dir = "./checkpoints"
        num_episodes = 10
        max_steps = 1000


    config = Config()

    env = UnityEnv("Tennis")

    run(config, env)
