import torch
from torch.optim import Adam

from model import PolicyModel, ValueModel
from noise import OUNoise
from utils import hard_update, to_tensor, soft_update


class DDPGAgent(object):

    def __init__(self, i_agent, actor_input_shape, actor_output_shape, critic_input_shape,
                 hidden_units=(256, 128, 64,), lr_actor=1e-4, lr_critic=1e-3):
        self.i_agent = i_agent
        self.actor = PolicyModel(actor_input_shape, actor_output_shape,
                                 hidden_units=hidden_units)
        self.critic = ValueModel(critic_input_shape, hidden_units=hidden_units)
        self.target_actor = PolicyModel(actor_input_shape, actor_output_shape,
                                        hidden_units=hidden_units)
        self.target_critic = ValueModel(critic_input_shape, hidden_units=hidden_units)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

        self.noise = OUNoise(actor_output_shape)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def reset_noise(self):
        self.noise.reset()

    def step(self, state, add_noise=False, noise_scale=1.0):
        action = self.actor(state)

        if add_noise:
            action += to_tensor(self.noise.noise() * noise_scale)
        action = action.clamp(-1, 1)

        return action

    def update(self, tau):
        soft_update(self.target_actor, self.actor, tau)
        soft_update(self.target_critic, self.critic, tau)

    def train_mode(self):
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

    def save(self, file_prefix):
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict()
        }
        torch.save(save_dict, self._model_filename(file_prefix))

    def load(self, file_prefix):
        save_dict = torch.load(self._model_filename(file_prefix))
        self.actor.load_state_dict(save_dict['actor'])
        self.critic.load_state_dict(save_dict['critic'])
        self.target_actor.load_state_dict(save_dict['target_actor'])
        self.target_critic.load_state_dict(save_dict['target_critic'])

    def _model_filename(self, file_prefix):
        return "%s.actor-%02d.pth" % (file_prefix, self.i_agent)
