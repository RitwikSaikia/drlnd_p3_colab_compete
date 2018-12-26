from sys import platform

import numpy as np
from gym.spaces import Box, Discrete
from unityagents import UnityEnvironment


def unity_filename(env_name):
    if platform == "linux" or platform == "linux2":
        env_filename = 'envs/%s_Linux_NoVis/%s.x86_64' % (env_name, env_name)
    elif platform == "darwin":
        env_filename = 'envs/%s.app' % env_name
    elif platform == "win32":
        env_filename = 'envs/%s_Windows_x86_64/%s.exe' % (env_name, env_name)

    return env_filename


class UnityEnv:

    def __init__(self, env_name, **kwargs) -> None:
        super().__init__()

        filename = unity_filename(env_name)
        self.unity_env = UnityEnvironment(file_name=filename, **kwargs)
        self.brain_names = self.unity_env.brain_names
        self.name = env_name

        self.state_space = []
        self.action_space = []
        self.num_agents_per_brain = []
        self.num_agents = 0

        for brain_name in self.brain_names:
            brain = self.unity_env.brains[brain_name]

            env_info = self.unity_env.reset(train_mode=True)[brain_name]

            num_agents = len(env_info.agents)
            num_actions = brain.vector_action_space_size
            states = env_info.vector_observations
            num_states = states.shape[1]

            self.num_agents += num_agents

            self.num_agents_per_brain.append(num_agents)

            for i in range(num_agents):
                action_space_type = brain.vector_action_space_type
                if action_space_type == 'continuous':
                    self.action_space.append(Box(-1, 1, shape=(num_actions,), dtype=np.float32))
                elif action_space_type == 'discrete':
                    self.action_space.append(Discrete(num_actions))
                else:
                    raise Exception("Unsupported action space type: %s" % action_space_type)

                observation_space_type = brain.vector_observation_space_type
                if observation_space_type == 'continuous':
                    self.state_space.append(Box(-10, 10, shape=(num_states,), dtype=np.float32))
                elif observation_space_type == 'discrete':
                    self.state_space.append(Discrete(num_states))
                else:
                    raise Exception("Unsupported observation space type: %s" % observation_space_type)

    def reset(self, train_mode=True):
        state = []
        for brain_name in self.brain_names:
            env_info = self.unity_env.reset(train_mode=train_mode)[brain_name]
            state.append(env_info.vector_observations)
        return np.vstack(state)

    def step(self, actions):
        next_states = []
        rewards = []
        dones = []

        step_input = {}
        offset = 0
        for b, brain_name in enumerate(self.brain_names):
            brain_actions = []
            for a in range(self.num_agents_per_brain[b]):
                agent_action = actions[offset + a]
                if isinstance(self.action_space[offset + b], Discrete):
                    agent_action = np.argmax(agent_action)
                brain_actions.append(agent_action)

            step_input[brain_name] = brain_actions
            offset += b

        env_infos = self.unity_env.step(step_input)

        for b, brain_name in enumerate(self.brain_names):
            env_info = env_infos[brain_name]
            brain_next_states = env_info.vector_observations
            brain_rewards = env_info.rewards
            brain_dones = env_info.local_done

            next_states.append(brain_next_states)
            rewards.append(brain_rewards)
            dones.append(brain_dones)

        next_states = np.vstack(next_states)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)

        return np.asarray(next_states), np.asarray(rewards), np.asarray(dones), env_infos

    def render(self, **kwargs):
        pass

    def close(self):
        pass
