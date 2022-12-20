from nets.acml_networks import Critic, CoordNet, MGenerator, Actor,  Attention
from algorithms.AgentBaseNet import *


class ACML(AgentBaseNet):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, dim_obs, dim_thought, dim_act, max_agents, nagents= 3, gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False, cuda=True):
        '''
        self.exploration = [OUNoise(self.dim_act) for i in range(self.nagents)]
        self.model_list = []
        self.model_name_list = []
        self.opt_list=[]
        self.opt_name_list = []
        '''

        AgentBaseNet.__init__(self, dim_act, nagents, gamma,
                              tau, lr, hidden_dim, discrete_action, cuda)

        self.dim_obs = dim_obs
        self.dim_thought = dim_thought
        self.max_links = max_agents


        dim_dummy = 3
        self.dim_nearby = 3
        dim_hidden = 64
        dim_joint_obs = 4
        dim_adj_landmarks = 4

        self.mgenerator = MGenerator(dim_obs, dim_thought)
        self.actor = [Actor(dim_obs+dim_thought, dim_act) for i in range(nagents)]
        self.critic = Critic(dim_obs, dim_act, nagents)
        self.comm = CoordNet(dim_thought)
        self.attention = Attention(dim_obs)


        self.mgenerator_target = MGenerator(dim_obs, dim_thought)
        self.actor_target = [Actor(dim_obs+dim_thought, dim_act) for i in range(nagents)]
        self.critic_target = Critic(dim_obs, dim_act, nagents)
        self.comm_target = CoordNet(dim_thought)


        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.actor_optimizers = [torch.optim.Adam(itertools.chain(self.mgenerator.parameters(),
                             self.actor[i].parameters(),self.comm.parameters(),self.attention.parameters()), lr=self.lr) for i in range(nagents)]


        self.model_list = [self.mgenerator,self.actor,self.critic,self.comm,self.attention,
                           self.mgenerator_target,self.actor_target,self.critic_target,self.comm_target,]
        self.model_name_list = ['MGenerator','actor','critic','comm','attention',
                                'MGenerator_target','Actor_target','critic_target','comm_target']
        self.opt_list = [self.actor_optimizers,self.critic_optimizer]
        self.opt_name_list = ['actor_optimizer','critic_optimizer']


        hard_update(self.mgenerator, self.mgenerator_target)
        hard_update_lst(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)
        hard_update(self.comm, self.comm_target)

        # moving all models to the given devicesF
        if cuda:
            self.mov_all_models('cuda')
        else:
            self.mov_all_models('cpu')

    '''
    input: obs
    output: thought_after
    '''
    def get_comm_result(self, obs, targ_flag=False):

        messages = self.mgenerator(obs)
        weights = self.attention(obs)
        #messages = messages*weights

        if targ_flag:
            messages_after = self.comm_target(messages)
        else:
            messages_after = self.comm(messages)
        weights = torch.cat((weights, 1.0-weights), dim=-1)
        return messages_after, weights


    def get_rollout_action(self, obs):
        thought_after, weights_prob = self.get_comm_result(obs)
        # thought[1 - agent_not_com_flag.repeat(1,1,self.dim_thought)] = 0.0
        action = torch.cat(
            tuple(self.actor[i](obs[i, :], thought_after[i, :]).unsqueeze(0) for i in range(obs.shape[0])), 0)
        return action

    def get_target_action(self, obs):
        thought_after,weights_prob = self.get_comm_result(obs, targ_flag=True)
        # ★★ ★★  此处获取thought after 是zeros的agent和batch，把他们选出来
        #thought[1-agent_not_com_flag.repeat(1,1,self.dim_thought)] = 0.0
        # 单独计算每个agent的action
        action = torch.cat(tuple(self.actor_target[i](obs[i,:],thought_after[i,:]).unsqueeze(0) for i in range(obs.shape[0])),0)
        return action, weights_prob

    def get_action(self, obs):
        thought_after,weights_prob = self.get_comm_result(obs)
        #thought[1 - agent_not_com_flag.repeat(1,1,self.dim_thought)] = 0.0
        action = torch.cat(tuple(self.actor[i](obs[i,:],thought_after[i, :]).unsqueeze(0) for i in range(obs.shape[0])), 0)
        return action, weights_prob

    def update(self, sample, logger=None, time_update=0):
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
        obs, acs, rews, next_obs, dones= sample
        obs = torch.stack(obs)
        acs = torch.stack(acs)
        rews = torch.stack(rews)
        next_obs = torch.stack(next_obs)
        dones = torch.stack(dones)

        self.critic_optimizer.zero_grad()
        # 更新critic的值
        # 此时假设下一步的comm_topology是不变的
        # dimensions are n_agents * n_batch * n_dim
        all_trgt_acs, _ = self.get_target_action(next_obs)
        if self.discrete_action: # one-hot encode action
            all_trgt_acs = onehot_from_logits(all_trgt_acs)
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=-1)

        target_value = (rews[0].view(-1, 1) + self.gamma *
                        self.critic_target(trgt_vf_in) *
                        (1 - dones[0].view(-1, 1)))

        vf_in = torch.cat((*obs, *acs), dim=-1)
        actual_value = self.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        for i in range(obs.shape[0]):
            self.actor_optimizers[i].zero_grad()

            #curr_pol_out, weights_prob = self.get_action_i(obs, agents_nearby, comm_top,i)
            curr_pol_out, weights_prob = self.get_action(obs)
            if self.discrete_action:
                curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
            else:
                curr_pol_vf_in = curr_pol_out
            vf_in = torch.cat((*obs, *curr_pol_vf_in), dim=-1)
            pol_loss = -self.critic(vf_in).mean()
            pol_loss += (curr_pol_vf_in**2).mean() * 1e-2
            pol_loss += torch.distributions.Categorical(probs = weights_prob.squeeze(-1)).entropy().mean()
            pol_loss.backward()

            self.actor_optimizers[i].step()

        # if logger is not None:
        #     self.add_log(logger, {'vf_loss': vf_loss,
        #                         'pol_loss': pol_loss})
        #     self.niter += 1

    def update_all_targets(self):
        soft_update(self.comm_target, self.comm, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.mgenerator_target, self.mgenerator, self.tau)
        soft_update_lst(self.actor_target, self.actor, self.tau)