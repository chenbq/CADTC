from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from nets.networks import MLPNetwork,GaussianNet,DiscretNet,MLPOMNetwork
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.noise import OUNoise
import torch.nn as nn
import torch




class BaseACAgent(object):
    """
        General class for DDPG agents (policy, critic, target policy, target
        critic, exploration noise)
        """

    def __init__(self, ):
        pass

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        # print(obs.device)
        action = self.policy(obs)
        if action.device != 'cpu':
            action = action.to('cpu')
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
            # action = action.clamp(0, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        for state in self.policy_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to('cuda')
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        for state in self.critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to('cuda')

    # including optimizer
    def mov_all_models(self, device='cpu'):
        self.policy.to(device)
        self.critic.to(device)
        self.target_policy.to(device)
        self.target_critic.to(device)
        for state in self.policy_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to(device)
        for state in self.critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to(device)


class DDPGAgent(BaseACAgent):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,hidden_dim_critic=64,
                 lr=0.01, discrete_action=True, cuda=True):
        BaseACAgent.__init__(self)
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim_critic,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim_critic,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action
        self.cuda=cuda

        if cuda:
            self.mov_all_models('cuda')
        else:
            self.mov_all_models('cpu')

class OMDDPGAgent(BaseACAgent):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=256,hidden_dim_critic=256,
                 lr=0.01, discrete_action=True, cuda=True):
        BaseACAgent.__init__(self)
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPOMNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim_critic,
                                 constrain_out=False)
        self.target_policy = MLPOMNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim_critic,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action
        self.cuda=cuda

        if cuda:
            self.mov_all_models('cuda')
        else:
            self.mov_all_models('cpu')


class GaussianAgent(BaseACAgent):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, num_vf_in, action_scale = 1.0, action_bias = 0.0,
                 hidden_dim=256,hidden_dim_critic=256,
                 lr=0.01, discrete_action=True, cuda=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        BaseACAgent.__init__(self)
        self.policy = GaussianNet(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim, action_scale = action_scale, action_bias = action_bias)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim_critic,
                                 constrain_out=False)

        self.value_func = MLPNetwork(num_vf_in, 1,
                                 hidden_dim=hidden_dim_critic,
                                 constrain_out=False)

        self.target_policy = GaussianNet(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim, action_scale = action_scale, action_bias = action_bias)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim_critic,
                                        constrain_out=False)
        self.target_value_func = MLPNetwork(num_vf_in, 1,
                                     hidden_dim=hidden_dim_critic,
                                     constrain_out=False)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        hard_update(self.target_value_func, self.value_func)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.value_func_optimizer = Adam(self.value_func.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action
        self.cuda=cuda

        if cuda:
            self.mov_all_models('cuda')
        else:
            self.mov_all_models('cpu')



    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'value_func': self.value_func.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'target_value_func': self.target_value_func.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'value_func_optimizer': self.value_func_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.value_func.load_state_dict(params['value_func'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.target_value_func.load_state_dict(params['target_value_func'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        for state in self.policy_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to('cuda')
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        for state in self.critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to('cuda')
        self.value_func_optimizer.load_state_dict(params['value_func_optimizer'])
        for state in self.value_func_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to('cuda')

    # including optimizer
    def mov_all_models(self, device='cpu'):
        self.policy.to(device)
        self.critic.to(device)
        self.value_func.to(device)
        self.target_policy.to(device)
        self.target_critic.to(device)
        self.target_value_func.to(device)
        for state in self.policy_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to(device)
        for state in self.critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to(device)
        for state in self.value_func_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to(device)


class DiscreteAgent(BaseACAgent):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, num_vf_in, action_scale = 1.0, action_bias = 0.0,
                 hidden_dim=256,hidden_dim_critic=256,
                 lr=0.01, discrete_action=True, cuda=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        BaseACAgent.__init__(self)
        self.policy = DiscretNet(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim_critic,
                                 constrain_out=False)

        self.value_func = MLPNetwork(num_vf_in, 1,
                                 hidden_dim=hidden_dim_critic,
                                 constrain_out=False)

        self.target_policy = DiscretNet(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim_critic,
                                        constrain_out=False)
        self.target_value_func = MLPNetwork(num_vf_in, 1,
                                     hidden_dim=hidden_dim_critic,
                                     constrain_out=False)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        hard_update(self.target_value_func, self.value_func)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.value_func_optimizer = Adam(self.value_func.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action
        self.cuda=cuda

        if cuda:
            self.mov_all_models('cuda')
        else:
            self.mov_all_models('cpu')



    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'value_func': self.value_func.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'target_value_func': self.target_value_func.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'value_func_optimizer': self.value_func_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.value_func.load_state_dict(params['value_func'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.target_value_func.load_state_dict(params['target_value_func'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        for state in self.policy_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to('cuda')
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        for state in self.critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to('cuda')
        self.value_func_optimizer.load_state_dict(params['value_func_optimizer'])
        for state in self.value_func_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to('cuda')

    # including optimizer
    def mov_all_models(self, device='cpu'):
        self.policy.to(device)
        self.critic.to(device)
        self.value_func.to(device)
        self.target_policy.to(device)
        self.target_critic.to(device)
        self.target_value_func.to(device)
        for state in self.policy_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to(device)
        for state in self.critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to(device)
        for state in self.value_func_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor) and self.cuda:
                    state[k] = v.to(device)