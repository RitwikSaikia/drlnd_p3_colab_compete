import functools

import torch
from torch.nn import MSELoss

from ddpg_agent import DDPGAgent
from replay_buffer import ReplayBuffer


class MADDPGAgent(object):

    def __init__(self, env, buffer_size=int(1e6), gamma=0.95, tau=0.01):
        self.num_agents = env.num_agents

        self.buffer = ReplayBuffer(buffer_size, self.num_agents)
        self.gamma = gamma
        self.tau = tau

        self.agents = []
        for action_space, state_space in zip(env.action_space, env.state_space):
            actor_input_shape = state_space.shape[0]
            actor_output_shape = action_space.shape[0]

            critic_input_shape = 0
            for space in [env.state_space, env.action_space]:
                critic_input_shape += functools.reduce(lambda a, b: a.shape[0] + b.shape[0], space)

            agent = DDPGAgent(len(self.agents), actor_input_shape, actor_output_shape, critic_input_shape)
            self.agents.append(agent)

    def reset_noise(self):
        for agent in self.agents:
            agent.reset_noise()

    def step(self, states, add_noise=False, noise_scale=1.0):
        return [a.step(state, add_noise=add_noise, noise_scale=noise_scale) for a, state in zip(self.agents, states)]

    def learn(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        self.train_mode()
        for i_agent in range(self.num_agents):
            sample = self.buffer.sample(batch_size)
            self._update(sample, i_agent)

        self._update_all_targets()
        self.eval_mode()

    def remember(self, states, agent_actions, rewards, next_states, dones):
        self.buffer.append(states, agent_actions, rewards, next_states, dones)

    def _update(self, experience, i_agent):
        states, actions, rewards, next_states, dones = experience
        curr_agent = self.agents[i_agent]

        curr_agent.critic_optimizer.zero_grad()

        all_target_actions = [actor(next_state) for actor, next_state in zip(self._target_actors, next_states)]

        target_critic_input = torch.cat((*next_states, *all_target_actions), dim=1)

        target_value = rewards[i_agent].view(-1, 1) \
                       + self.gamma * curr_agent.target_critic(target_critic_input) * (1 - dones[i_agent].view(-1, 1))

        critic_input = torch.cat((*states, *actions), dim=1)
        estimated_value = curr_agent.critic(critic_input)
        critic_loss = MSELoss()(estimated_value, target_value.detach())

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.actor_optimizer.zero_grad()

        curr_actor_action = curr_agent.actor(states[i_agent])

        all_actor_actions = []
        for i, actor, state in zip(range(self.num_agents), self._actors, states):
            if i == i_agent:
                all_actor_actions.append(curr_actor_action)
            else:
                all_actor_actions.append(actor(state))

        critic_input = torch.cat((*states, *all_actor_actions), dim=1)

        actor_loss = -curr_agent.critic(critic_input).mean()
        actor_loss += (curr_actor_action ** 2).mean() * 1e-3

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 0.5)
        curr_agent.actor_optimizer.step()

    def _update_all_targets(self):
        for agent in self.agents:
            agent.update(self.tau)

    @property
    def _actors(self):
        return [agent.actor for agent in self.agents]

    @property
    def _target_actors(self):
        return [agent.target_actor for agent in self.agents]

    def train_mode(self):
        for agent in self.agents:
            agent.train_mode()

    def eval_mode(self):
        for agent in self.agents:
            agent.eval_mode()

    def save(self, file_prefix):
        self.train_mode()
        for agent in self.agents:
            agent.save(file_prefix)

    def load(self, file_prefix):
        for agent in self.agents:
            agent.load(file_prefix)
