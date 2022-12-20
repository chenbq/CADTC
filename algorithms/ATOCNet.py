from nets.atoc_networks import Critic, Actor_I, Actor_II, Comm, Attention
from algorithms.AgentBaseNet import *

class ATOCNet(AgentBaseNet):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, dim_obs, dim_thought, dim_act, max_agents, nagents= 3, gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, cuda=True,
                 discrete_action=False):
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

        # 构建整体agent的network
        self.actor_I = Actor_I(dim_obs, dim_thought)
        self.actor_II = Actor_II(dim_thought, dim_act)
        self.critic = Critic(dim_obs, dim_act)
        self.attention = Attention(dim_thought)
        self.comm = Comm(nagents, dim_thought)

        self.actor_I_target = Actor_I(dim_obs, dim_thought)
        self.actor_II_target = Actor_II(dim_thought, dim_act)
        self.critic_target = Critic(dim_obs, dim_act)
        #self.attention_target = Attention(dim_thought)
        self.comm_target = Comm(nagents, dim_thought)

        #self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.actor_optimizer = torch.optim.Adam(itertools.chain(self.actor_I.parameters(),
                             self.actor_II.parameters()), lr=self.lr)
        self.comm_optimizer = torch.optim.Adam(self.comm.parameters(), lr=self.lr)
        self.attention_optimizer = torch.optim.Adam(self.attention.parameters(), lr=self.lr)

        self.model_list = [self.actor_I, self.actor_II, self.critic, self.comm, self.attention,
                           self.actor_I_target, self.actor_II_target, self.critic_target, self.comm_target]
        self.model_name_list = ['actor_I', 'actor_II', 'critic', 'comm', 'attention',
                                'actor_I_target', 'actor_II_target', 'critic_target', 'comm_target']
        self.opt_list = [self.actor_optimizer, self.critic_optimizer,
                         self.comm_optimizer, self.attention_optimizer,]
        self.opt_name_list = ['actor_optimizer', 'critic_optimizer',
                              'comm_optimizer', 'attention_optimizer']

        hard_update(self.actor_I, self.actor_I_target)
        hard_update(self.actor_II, self.actor_II_target)
        hard_update(self.critic, self.critic_target)
        #hard_update(self.attention, self.attention_target)
        hard_update(self.comm, self.comm_target)

        # moving all models to the given devicesF
        if cuda:
            self.mov_all_models('cuda')
        else:
            self.mov_all_models('cpu')

        #对于ATOC是额外的exploration项目
        self.att_exploration = [OUNoise(1) for i in range(self.nagents)]

    def reset_noise(self):
        if not self.discrete_action:
            [self.exploration[i].reset() for i in range(self.nagents)]
        [self.att_exploration[i].reset() for i in range(self.nagents)]

    def scale_noise(self, scale):
        for i in range(self.nagents):
            if self.discrete_action:
                self.exploration[i] = scale
            else:
                self.exploration[i].scale = scale
            self.att_exploration[i].scale = 0.5
        # (n_agents, dim_thought)

    def get_thought(self, obs):
        thought = self.actor_I(obs)
        return thought

    def get_head_flag(self, thought, explore=True, noise_ratio = 0.3):
        head_flag = self.attention(thought)
        '''for agent_idx in range(self.nagents):
            head_flag[agent_idx, :, :] += Tensor((np.random.rand(1,1)*2-1)*np.exp(-5*self.episode_num))

                #Variable(Tensor(self.att_exploration[agent_idx].noise()),
                                            #requires_grad=False)

        #action[agent_idx, :, :] += Variable(Tensor(self.exploration.noise()),
        #                                   requires_grad=False)
        #head_flag += Variable(Tensor(np.random.randn(head_flag.shape[0],head_flag.shape[1],head_flag.shape[2])),
                                           #requires_grad=False)
        head_flag = head_flag.clamp(0, 1)'''
        return head_flag

    #　thought 维度为num_agent, num_batch, dim_thought
    # comm_top记录的是通信拓扑，如果agent_i与任意agent_j没有联系，那么comm_top(i,i)=0
    # 否则，comm_top(i,i)=1

    # 灵活运用矩阵slice x[x>0.5]
    # reshape 重新生成矩阵
    # repeat
    # 转置 .T
    '''
    获取通信后的结果
    '''
    def get_comm_result(self, thought, comm_top, mode = 'acting', targ_flag=False):
        # 此处加入detach，是为了独立训练thought和comm的生成，单独训练，确保空间较小
        thought_after = thought.clone().detach()
        thought_after = thought_after.permute(1,0,2)
        agents_comm_flag = torch.ones_like(thought[:, :, 0], dtype=torch.uint8)
        for i in range(self.nagents):
            if (comm_top[i]==1).sum()==0:
                #thought_after[i,:,:] = torch.zeros_like(thought[0,:,:])
                continue
            # shape (batch_size, n_agents)
            idx_mat =  comm_top[i]==1
            agents_comm_flag[idx_mat.t()] = 0
            # 判断idx_mat不全为false的列，也就是有几个batch不为空。
            # 转置， 增加最后一个维度，重复dim_thought次。
            # 此时shape (n_agents, batch_size_changed, dim_thought)
            #idx_mat_t = idx_mat.t().reshape(self.nagents,-1,1).repeat(1,1,self.dim_thought)
            #thought_pre_i = thought_after[idx_mat_t].reshape(self.max_links,-1,self.dim_thought)
            #thought_after_i = self.comm(thought_pre_i)
            #thought_after[idx_mat_t] = thought_after_i.reshape(-1)

            # 此处要保持batch_size， agent的纬度，因为每一个batch_size都有固定数的agent参与，但是反之则不是，不能保证重排时的结构正确。
            idx_mat_t = idx_mat.reshape(-1,self.nagents,1).repeat(1,1,self.dim_thought)
            # 此处对thought_after访问前需要重排thought_after的结构，前两纬度交换顺序
            #thought_after_temp = thought_after.permute(1,0,2)
            # 此时得到的thought_pre_i还是batch_size， agent的纬度
            thought_pre_i = thought_after[idx_mat_t].reshape(-1,self.max_links,self.dim_thought)
            thought_pre_i = thought_pre_i.permute(1,0,2)
            #此时输出的纬度是agent，batch_size
            if targ_flag:
                #thought_after_i = self.comm_target(thought_pre_i.clone().detach())
                thought_after_i = self.comm_target(thought_pre_i)
            else:
                #thought_after_i = self.comm(thought_pre_i.clone().detach())
                thought_after_i = self.comm(thought_pre_i)
            thought_after_i = thought_after_i.permute(1,0,2)
            thought_after[idx_mat_t] = thought_after_i.reshape(-1)


        # 在最后，将未参与comm的thought_after置为零。
        agent_not_com_flag = agents_comm_flag.reshape(self.nagents, -1, 1)
        agents_comm_flag = agents_comm_flag.reshape(self.nagents, -1, 1).repeat(1, 1, self.dim_thought)
        thought_after = thought_after.permute(1, 0, 2)
        thought_after[agents_comm_flag] = 0.0
        return thought_after, agent_not_com_flag

    # 获取每一个agent的本身thought，以及通信后的d_Q的值，放到一个队列中去。
    # 在get comm后的thought的同时，获取dQ的信息，
    # 此处可能要多调用几次，actor_II的模块。
    # 此处在有多个batch/parallel的env输入的时候，存在错误。
    # 2020-12-17 此处改成采样和执行两个阶段，采样后，保留大于零的部分进行通信。
    def get_comm_dQ_result(self, obs, nearby_agents, comm_top=None, mode='initial', com_flag=False, explore_flag=True):

        thought = self.get_thought(obs)
        threshold_com = 0.5

        if True:
            # explore_flag:
            # 计算出d_Q大于零的数值
            comm_top_tmp = Tensor(np.array(nearby_agents))
            # head_flag = torch.zeros_like(head_flag)
            # 每个initiator计算d_Q
            thought_temp = thought.clone()
            thought_attention_list = []
            dq_attention_list = []
            for i in range(self.nagents):
                idx_mat = comm_top_tmp[i] == 1
                # 判断idx_mat不全为false的列，也就是有几个batch不为空。
                # 转置， 增加最后一个维度，重复dim_thought次。
                # 此时shape (n_agents, batch_size_changed, dim_thought)
                idx_mat_t = idx_mat.t().reshape(self.nagents, -1, 1).repeat(1, 1, self.dim_thought)
                idx_mat_o = idx_mat.t().reshape(self.nagents, -1, 1).repeat(1, 1, self.dim_obs)
                '''至此，得到了thought中参与的agent和batch的列表'''

                thought_pre_i = thought_temp[idx_mat_t].reshape(self.max_links, -1, self.dim_thought)
                thought_after_i = self.comm(thought_pre_i)
                obs_i = obs[idx_mat_o].reshape(self.max_links, -1, self.dim_obs)

                # d_Q = self.critic(obs_i, self.actor_II(thought_pre_i, thought_after_i),thought_after_i).sum() - \
                #      self.critic(obs_i, self.actor_II(thought_pre_i, torch.zeros_like(thought_pre_i)),torch.zeros_like(thought_pre_i)).sum()
                d_Q = self.critic(obs_i, self.actor_II(torch.zeros_like(thought_pre_i), thought_after_i)).sum() - \
                      self.critic(obs_i, self.actor_II(thought_pre_i, torch.zeros_like(thought_pre_i))).sum()
                # if d_Q>0:
                # head_flag[i,:,:] = 1
                dq_attention_list.append(d_Q)
                # 此处是否应该保存此前的thought，虽然是此前的thought带来的attention模块，使得建立intiator，
                # 但是此处的d_Q是基于thought变化后的改变量带来的。
                # 所以后续要尝试，此处采用thought_pre_i的当前initiator的thought来表示
                thought_attention_list.append(thought_temp[i, 0, :])  # test adding though_after value
            '''if (head_flag > threshold_com).sum() <= 1: # or (head_flag > threshold_com).sum() == self.nagents:
                num_initiator = np.random.randint(3, 5)
                # threshold_com = head_flag.mean()
                head_flag[:num_initiator, :, :] = 1
                head_flag[num_initiator:, :, :] = 0
            head_flag = (head_flag >= threshold_com)'''

        if mode == 'initial':
            head_flag = self.get_head_flag(thought)

            '''#此步骤的判断是为了增加exploration，避免变化过快。
            if (head_flag>0.5).sum()==0 or (head_flag>0.5).sum()==self.nagents:
                num_initiator = np.random.randint(2,6)
                #threshold_com = head_flag.mean()
                head_flag[:num_initiator,:,:]=1
                head_flag[num_initiator:,:,:] = 0
            #threshold_com = 0.5
            #threshold_com = 2
            if com_flag:
                threshold_com = 0
            else:
                threshold_com = 2
            #threshold_com = 0 # first try revising all to one
            #threshold_com = 2 # second try revising all to zero, i.e., forbbiden comm
            '''

            # sample head_flag with probabilistic comm --> 12-22 not very effective
            # head_flag = (head_flag-torch.rand(head_flag.shape))>0
            if head_flag.device != 'cpu':
                ord_list = np.argsort(head_flag.cpu().detach().numpy()[:, 0, 0])[::-1]
            else:
                ord_list = np.argsort(head_flag.detach().numpy()[:, 0, 0])[::-1]
            head_flag = head_flag >= threshold_com
            if (head_flag > threshold_com).sum() <= 1:  # or (head_flag > threshold_com).sum() == self.nagents:
                # num_initiator = np.random.randint(2, 4)
                # threshold_com = head_flag.mean()
                # head_flag[ord_list[:num_initiator].tolist(), :, :] = 1
                # head_flag[ord_list[num_initiator:].tolist(), :, :] = 0
                # 若不通信改为固定的
                head_flag[[0, 3, 6], :, :] = 1
                head_flag[[1, 2, 4, 5], :, :] = 0
            # Force to change head_flag
            head_flag[[0, 3, 6], :, :] = 1
            head_flag[[1, 2, 4, 5], :, :] = 0
            # remove redundant oprtation
            # head_flag = (head_flag >= threshold_com)

            head_flag = head_flag.repeat(1, 1, self.nagents)
            # 此后要根据head_flag计算通信的列表。
            # ci
            # comm_top = torch.from_numpy(np.array(nearby_agents)) * head_flag
            # 判断是否两个agent通信拓扑相同，避免无效通信
            # 12-18 允许多次参与initiator

            comm_top = np.array(nearby_agents) * head_flag
            already_com_set = []
            for agenti in range(self.nagents):
                if comm_top.device != 'cpu':
                    curr_top = comm_top[agenti, 0, :].cpu().detach().numpy()
                else:
                    curr_top = comm_top[agenti, 0, :].detach().numpy()
                if any((curr_top == x).all() for x in already_com_set):
                    comm_top[agenti, 0, :] = 0
                else:
                    already_com_set.append(curr_top)

        thought_after = thought.clone()

        # thought_attention_list = []
        # dq_attention_list = []
        agents_comm_flag = torch.ones_like(thought[:, :, 0], dtype=torch.uint8)
        comm_counts = np.zeros_like(comm_top[0, :, :])
        for i in range(self.nagents):

            # 对于每一个initiator，都要算一遍本身的thought的action和更改后的thought的action
            # if (comm_top[i]==1).sum()==0:
            #    thought_after[i,:,:] = torch.zeros_like(thought[0,:,:])
            #    continue
            # shape (batch_size, n_agents)
            if (comm_top[i] == 1).sum() == 0:
                continue
            idx_mat = comm_top[i] == 1
            agents_comm_flag[idx_mat.t()] = 0

            # 判断idx_mat不全为false的列，也就是有几个batch不为空。
            # 转置， 增加最后一个维度，重复dim_thought次。
            # 此时shape (n_agents, batch_size_changed, dim_thought)
            idx_mat_t = idx_mat.t().reshape(self.nagents, -1, 1).repeat(1, 1, self.dim_thought)
            idx_mat_o = idx_mat.t().reshape(self.nagents, -1, 1).repeat(1, 1, self.dim_obs)
            '''至此，得到了thought中参与的agent和batch的列表'''

            thought_pre_i = thought_after[idx_mat_t].reshape(self.max_links, -1, self.dim_thought)
            thought_ori_i = thought[idx_mat_t].reshape(self.max_links, -1, self.dim_thought)
            thought_after_i = self.comm(thought_pre_i)
            obs_i = obs[idx_mat_o].reshape(self.max_links, -1, self.dim_obs)

            '''此处采样挪到上面专门采样的地方'''
            '''d_Q = self.critic(obs_i, self.actor_II(thought_pre_i, thought_after_i)).sum() - \
            self.critic(obs_i, self.actor_II(thought_ori_i, torch.zeros_like(thought_pre_i))).sum()
            # 此时，得到了comm前后的thought，可以计算当前的critic差值。

            #对于不是首次通信的agent，不添加thought
            if True: #comm_counts[0,i]==0:
                dq_attention_list.append(d_Q)
                # 此处是否应该保存此前的thought，虽然是此前的thought带来的attention模块，使得建立intiator，
                # 但是此处的d_Q是基于thought变化后的改变量带来的。
                # 所以后续要尝试，此处采用thought_pre_i的当前initiator的thought来表示
                thought_attention_list.append(thought[i,0,:]) #test adding though_after value
                #thought_attention_list.append(thought_after[i,0,:]) #test adding though_after value'''

            thought_after[idx_mat_t] = thought_after_i.reshape(-1)
            comm_counts += comm_top[i]

        # 在最后，将未参与comm的thought_after置为零。
        # print(agents_comm_flag.sum())
        agents_comm_flag = agents_comm_flag.reshape(self.nagents, -1, 1).repeat(1, 1, self.dim_thought)
        thought_after[agents_comm_flag] = 0.0
        # 将当前的thought中通信后的改为零。
        # pre_agents_comm_flag = 1 - agents_comm_flag
        # thought[pre_agents_comm_flag] = 0.0
        if np.max(dq_attention_list) == np.min(dq_attention_list):
            dq_attention_list = []
            thought_attention_list = []
        return thought, thought_after, thought_attention_list, dq_attention_list, comm_top

    # 此处还需要做更多修改

    def step(self, obs, nearby_agents, explore=False, com_mode = 'initial', comm_top_in=None, com_flag=False):
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

        thought, thought_after, thought_array, dQ_array, comm_top = self.get_comm_dQ_result(obs, nearby_agents,com_flag=com_flag,
                                                                                            comm_top=comm_top_in, mode=com_mode)
        action = self.actor_II(thought, thought_after)
        #d_Q = np.zeros(( self.nagents, thought.shape[1],  ))
        if action.device != 'cpu':
            action = action.to('cpu')
            thought_array = [ h.to('cpu') for h in thought_array]
            dQ_array =   [ h.to('cpu') for h in dQ_array]
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
        return action, thought_array, dQ_array, comm_top


    def get_target_action(self, obs, comm_top):
        thought = self.actor_I_target(obs)
        thought_after, agent_not_com_flag = self.get_comm_result(thought, comm_top, targ_flag=True)
        # ★★ ★★  此处获取thought after 是zeros的agent和batch，把他们选出来
        #thought[1-agent_not_com_flag.repeat(1,1,self.dim_thought)] = 0.0
        action = self.actor_II_target(thought, thought_after)
        return action, agent_not_com_flag

    def get_action(self, obs, comm_top):
        thought = self.actor_I(obs)
        thought_after, agent_not_com_flag = self.get_comm_result(thought, comm_top)
        #thought[1 - agent_not_com_flag.repeat(1,1,self.dim_thought)] = 0.0
        action = self.actor_II(thought, thought_after)
        return action, agent_not_com_flag




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
        obs, acs, rews, next_obs, dones, comm_top = sample
        obs = torch.stack(obs)
        acs = torch.stack(acs)
        rews = torch.stack(rews)
        next_obs = torch.stack(next_obs)
        dones = torch.stack(dones)
        comm_top = torch.stack(comm_top)

        # 更新critic的值
        # 此时假设下一步的comm_topology是不变的
        # dimensions are n_agents * n_batch * n_dim
        all_trgt_acs,  agent_not_com_flag = self.get_target_action(next_obs, comm_top)
        if self.discrete_action: # one-hot encode action
            all_trgt_acs = onehot_from_logits(all_trgt_acs)
        #trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        curr_pol_out, _ = self.get_action(obs, comm_top)
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic actor for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_pol_out



        if agent_not_com_flag.sum()!=0:
            # 没有通信的情况
            obs = obs[agent_not_com_flag.repeat(1,1,self.dim_obs)].reshape(-1,self.dim_obs)
            acs = acs[agent_not_com_flag.repeat(1,1,self.dim_act)].reshape(-1,self.dim_act)
            rews = rews[agent_not_com_flag.squeeze()].reshape(-1,1)
            next_obs = next_obs[agent_not_com_flag.repeat(1,1,self.dim_obs)].reshape(-1,self.dim_obs)
            dones = dones[agent_not_com_flag.squeeze()].reshape(-1,1)
            wo_trgt_acs = all_trgt_acs[agent_not_com_flag.repeat(1,1,self.dim_act)].reshape(-1,self.dim_act)
            wo_curr_pol_out = curr_pol_out[agent_not_com_flag.repeat(1,1,self.dim_act)].reshape(-1,self.dim_act)
            #wo_thought_a_targ = thought_a_targ[agent_not_com_flag.repeat(1,1,self.dim_thought)].reshape(-1,self.dim_thought)
            # ★★★ w/o communication
            self.critic_optimizer.zero_grad()

            target_value = (rews.view(-1, 1) + self.gamma *
                            self.critic_target(next_obs,wo_trgt_acs) *
                            (1 - dones.view(-1, 1)))


            #vf_in = torch.cat((*obs, *acs), dim=1)
            actual_value = self.critic(obs, acs)
            vf_loss = MSELoss(actual_value, target_value.detach())
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()



            self.actor_optimizer.zero_grad()


            pol_loss = -self.critic(obs, wo_curr_pol_out).mean()
            pol_loss += (wo_curr_pol_out**2).mean() * 1e-3
            pol_loss.backward()
            torch.nn.utils.clip_grad_norm(self.actor_I.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm(self.actor_II.parameters(), 0.5)
            #torch.nn.utils.clip_grad_norm(self.comm.parameters(), 0.25)
            self.actor_optimizer.step()
            #if True: #self.niter %10 ==0:
                #print('vf_loss:',vf_loss)
                #print('\n pol_loss:', pol_loss)
            # if logger is not None:
            #     logger.add_scalars('agent%i/losses' % 0,
            #                        {'no_com_vf_loss': vf_loss,
            #                         'no_com_pol_loss': pol_loss},
            #                        self.niter)
            #     self.niter += 1

            obs, acs, rews, next_obs, dones, comm_top = sample
            obs = torch.stack(obs)
            acs = torch.stack(acs)
            rews = torch.stack(rews)
            next_obs = torch.stack(next_obs)
            dones = torch.stack(dones)
            comm_top = torch.stack(comm_top)

        if (1-agent_not_com_flag.sum())!=0:
            curr_pol_out, _ = self.get_action(obs, comm_top)
            if self.discrete_action:
                # Forward pass as if onehot (hard=True) but backprop through a differentiable
                # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
                # through discrete categorical samples, but I'm not sure if that is
                # correct since it removes the assumption of a deterministic actor for
                # DDPG. Regardless, discrete policies don't seem to learn properly without it.
                curr_pol_out = gumbel_softmax(curr_pol_out, hard=True)
            else:
                curr_pol_out = curr_pol_out
            agent_not_com_flag = 1-agent_not_com_flag
            obs = obs[agent_not_com_flag.repeat(1, 1, self.dim_obs)].reshape(-1, self.dim_obs)
            acs = acs[agent_not_com_flag.repeat(1, 1, self.dim_act)].reshape(-1, self.dim_act)
            rews = rews[agent_not_com_flag.squeeze()].reshape(-1, 1)
            next_obs = next_obs[agent_not_com_flag.repeat(1, 1, self.dim_obs)].reshape(-1, self.dim_obs)
            dones = dones[agent_not_com_flag.squeeze()].reshape(-1, 1)
            wo_trgt_acs = all_trgt_acs[agent_not_com_flag.repeat(1, 1, self.dim_act)].reshape(-1, self.dim_act)
            wo_curr_pol_out = curr_pol_out[agent_not_com_flag.repeat(1, 1, self.dim_act)].reshape(-1, self.dim_act)
            #w_thought_a_targ = thought_a_targ[agent_not_com_flag.repeat(1, 1, self.dim_thought)].reshape(-1,                                                                                                          self.dim_thought)
            #w_thought_a = thought_a[agent_not_com_flag.repeat(1, 1, self.dim_thought)].reshape(-1,                                                                                                          self.dim_thought)
            # ★★★ w/ communication
            self.critic_optimizer.zero_grad()

            target_value = (rews.view( -1, 1) + self.gamma *
                            self.critic_target(next_obs, wo_trgt_acs) *
                            (1 - dones.view(-1, 1)))

            # vf_in = torch.cat((*obs, *acs), dim=1)
            actual_value = self.critic(obs, acs)
            vf_loss = MSELoss(actual_value, target_value.detach())
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            # 更新actor的值

            self.actor_optimizer.zero_grad()
            self.comm_optimizer.zero_grad()
            pol_loss = -self.critic(obs, wo_curr_pol_out).mean()
            pol_loss += (wo_curr_pol_out ** 2).mean() * 1e-3
            pol_loss.backward()
            torch.nn.utils.clip_grad_norm(self.actor_I.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm(self.actor_II.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm(self.comm.parameters(), 0.25)
            self.actor_optimizer.step()
            self.comm_optimizer.step()



            # if True: #self.niter %10 ==0:
            # print('vf_loss:',vf_loss)
            # print('\n pol_loss:', pol_loss)
            # if logger is not None:
            #     logger.add_scalars('agent%i/losses' % 0,
            #                        {'com_vf_loss': vf_loss,
            #                         'com_pol_loss': pol_loss},
            #                        self.niter)
            #     self.niter += 1


    def update_attention(self, sample):
        #2012-12-11 attention module without zero_grad, 无法提升
        thoughts, dQ = sample
        prob_lst = F.sigmoid(self.attention(thoughts))
        at_loss = torch.nn.BCELoss()(prob_lst, dQ)
        at_loss.backward()
        torch.nn.utils.clip_grad_norm(self.attention.parameters(), 0.5)
        self.attention_optimizer.step()
        return at_loss

    '''
    from utils.misc import plot_curve_with_label
    y=dQ.permute(1,0).clone().detach().numpy()
    y.sort()
    plot_curve_with_label(y,'a','b')
    z = prob_lst.permute(1,0).clone().detach().numpy()
    z.sort()
    plot_curve_with_label(z,'a','b')
    '''

    #11-18-2117: clip 改成了0.25
    #11-18-2118: update all 的时候，%10去掉了。
    def update_all_targets(self, time_update=0):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.comm_target, self.comm, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        #if time_update % 10 == 0:
        soft_update(self.actor_I_target, self.actor_I, self.tau)
        soft_update(self.actor_II_target, self.actor_II, self.tau)