#!/usr/bin/env python3

import logging
import os
import sys
from collections import deque

import numpy as np

from env import UnityEnv
from maddpg_agent import MADDPGAgent
from utils import set_seed, to_tensor

logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().setLevel(logging.DEBUG)


def run(config, env):
    checkpoints_dir = "%s/%s" % (config.checkpoints_dir, env.name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    maddpg = MADDPGAgent(env)

    all_scores = []
    scores = deque(maxlen=config.score_window)
    best_avg_score = -np.inf
    solved_score = None

    t = 0
    for i_episode in range(0, config.num_episodes):
        states = env.reset()
        maddpg.eval_mode()
        episode_rewards = np.zeros(maddpg.num_agents)

        exploration = max(0, config.num_exploration_episodes - i_episode) / config.num_exploration_episodes
        exploration = config.exploration_range[1] + (
                config.exploration_range[0] - config.exploration_range[1]) * exploration
        maddpg.reset_noise()

        for i_step in range(config.max_steps):
            torch_states = [to_tensor(states[i]) for i in range(maddpg.num_agents)]

            actions = maddpg.step(torch_states, add_noise=True, noise_scale=exploration)
            actions = [action.data.numpy() for action in actions]

            next_states, rewards, dones, _ = env.step(actions)
            maddpg.remember(states, actions, rewards, next_states, dones)
            states = next_states
            episode_rewards += rewards
            t += 1

            if (t % config.steps_per_update) == 0:
                maddpg.learn(config.batch_size)

            if np.any(dones):
                break

        score = np.max(episode_rewards)
        scores.append(score)
        all_scores.append(score)
        avg_score = np.mean(scores)
        np.savetxt("%s/scores.tsv" % checkpoints_dir, all_scores, fmt="%.5f")
        if len(scores) >= config.score_window and avg_score > best_avg_score:
            best_avg_score = avg_score
            maddpg.save("%s/model.best" % checkpoints_dir)

        logger.info("Episode %i/%i: eps: %.3f, Score: %.3f, Avg Score: %.3f, Best Avg Score: %.3f, Steps: %d"
                    % (i_episode + 1, config.num_episodes, exploration, score, avg_score, best_avg_score, i_step + 1))

        if solved_score is None and best_avg_score >= config.solve_score:
            solved_score = best_avg_score
            print("Solved in %i episodes" % (i_episode + 1))
            maddpg.save("%s/model.solved" % checkpoints_dir)

        sys.stdout.flush()

    np.savetxt("%s/scores.last.tsv" % checkpoints_dir, all_scores, fmt="%.5f")
    maddpg.save("%s/model.last" % checkpoints_dir)
    env.close()


if __name__ == '__main__':
    class Config:
        solve_score = 0.5
        score_window = 100
        seed = 0
        checkpoints_dir = "./checkpoints"
        num_episodes = 2000
        max_steps = 1000
        steps_per_update = 2
        batch_size = 128
        num_exploration_episodes = 25000
        exploration_range = (0.3, 0.0)


    config = Config()

    set_seed(config.seed)

    env = UnityEnv("Tennis", seed=config.seed)

    run(config, env)
