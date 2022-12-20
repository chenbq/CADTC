import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from gym.spaces import Box
from utils.misc import soft_update, hard_update, onehot_from_logits, gumbel_softmax, hard_update_lst, soft_update_lst
from utils.noise import OUNoise

import itertools
import torch
import numpy as np

MSELoss = torch.nn.MSELoss()

class AgentBaseNet(object):
    def __init__(self, dim_act = 2, nagents = 3, gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, discrete_action=False, cuda=True):

        '''
        :param dim_act:
        :param nagents:
        :param gamma:
        :param tau:
        :param lr:
        :param hidden_dim:
        :param discrete_action:
        :param cuda:
        '''

        self.nagents = nagents
        self.dim_act = dim_act

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.niter = 0
        self.cuda = cuda # move all models to the device
        self.discrete_action = discrete_action
        if not discrete_action:
            self.exploration = [OUNoise(self.dim_act) for i in range(self.nagents)]

        else:
            self.exploration =[0.3 for i in range(self.nagents)]

        self.model_list = []
        self.model_name_list = []
        self.opt_list=[]
        self.opt_name_list = []

    def reset_noise(self):
        if not self.discrete_action:
            [self.exploration[i].reset() for i in range(self.nagents)]

    def scale_noise(self, scale):
        for i in range(self.nagents):
            if self.discrete_action:
                self.exploration[i] = scale
            else:
                self.exploration[i].scale = scale

    def update_episode_num(self, num):
        self.episode_num = num
    def provide_env(self, env):
        self.env = env

    # 要以get_rollout_action为输入才可以
    def step(self, obs,  explore=False, return_weights = False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
            hidden_unit
            d_Q
        """
        if return_weights == True:
            weights_prob, action = self.get_rollout_action(obs, return_weights)
        else:
            action = self.get_rollout_action(obs)

        if action.device != 'cpu':
            action = action.to('cpu')
        for agent_idx in range(self.nagents):
            if self.discrete_action:
                if explore:
                    action[agent_idx,:,:] = gumbel_softmax(action[agent_idx,:,:], hard=True)
                else:
                    action[agent_idx,:,:] = onehot_from_logits(action[agent_idx,:,:])
            else:  # continuous action
                if explore:
                    action[agent_idx,:,:] += Variable(Tensor(self.exploration[agent_idx].noise()),
                                       requires_grad=False)
                action = action.clamp(-1, 1)

        if return_weights == True:
            return weights_prob, action
        else:
            return action

    def get_rollout_action(self, obs):
        pass

    def get_target_action(self, obs):
        pass

    def get_action(self, obs):
        pass


    def get_params(self):
        dict={}
        k_list = self.model_name_list + self.opt_name_list
        v_list = self.model_list + self.opt_list
        for pair in zip(k_list, v_list):
            if type(pair[1])==list:
                dict[pair[0]] = [x.state_dict() for x in pair[1]]
            else:
                dict[pair[0]] = pair[1].state_dict()
        return dict



    def load_params(self, params):
        k_list = self.model_name_list + self.opt_name_list
        v_list = self.model_list + self.opt_list

        for idx in range(len(k_list)):
            name = k_list[idx]
            model = v_list[idx]
            if type(params[name])==list:
                for i,v in enumerate(params[name]):
                    model[i].load_state_dict(v)
                    # if 'opt' in name and self.cuda:
                    #     self.move_optmizer_cuda(model[i])
            else:
                model.load_state_dict(params[name])
                # if 'opt' in name and self.cuda:
                #     self.move_optmizer_cuda(model)

    def update(self):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        pass

    def add_log(self, logger, dict_loss):
        if logger is not None:
            logger.add_scalars('agent%i/losses' % 0,
                               dict_loss,
                               self.niter)
            self.niter += 1

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        pass

    # moving all to given device, including optimizer
    def mov_all_models(self, device='cpu'):
        for model in self.model_list:
            if type(model) == list:
                for model_i in model:
                    model_i.to(device)
            else:
                model.to(device)
        for opt in self.opt_list:
            if type(opt) == list:
                for opt_i in opt:
                    self.move_optmizer_cuda(opt_i, device)
            else:
                self.move_optmizer_cuda(opt, device)

    def move_optmizer_cuda(self, optimizer, device):
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def prep_training(self):

        for model in self.model_list:
            if type(model) == list:
                for model_i in model:
                    model_i.train()
            else:
                model.train()

    def prep_rollouts(self):
        for model in self.model_list:
            if type(model) == list:
                for model_i in model:
                    model_i.eval()
            else:
                model.eval()

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.mov_all_models(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': self.get_params()}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, dim_thought = 128, gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, max_agents=3, cuda=True):
        """
        Instantiate instance of this class from multi-agent environment
        """
        adv_flag = env.agent_types.count('adversary')
        for type, acsp, obsp in zip(env.agent_types, env.action_space, env.observation_space):
            if adv_flag and type!='adversary':
                continue
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)

        #此处在num_agent限制，从而保证dim_thought的正确性
        if adv_flag:
            nagents = env.agent_types.count('adversary')
        else:
            nagents = env.agent_types.count('agent')

        init_dict = {'dim_obs': num_in_pol, 'dim_thought': dim_thought, 'dim_act': num_out_pol,
                     'max_agents': max_agents, 'nagents':nagents , 'gamma': gamma,
                     'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'discrete_action': discrete_action,
                     'cuda': cuda}
        instance = cls(**init_dict)

        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.load_params(save_dict['agent_params'])
        return instance