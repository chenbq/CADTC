import torch as th
import torch.nn as nn
import torch.nn.functional as F


# 通信链路输入为  (n_agent, batch, thought_dim)
# 输出为 (n_agent, batch, thought_dim):

class Comm(nn.Module):
    # max_agents为最大允许通信的agent数目，thought_dim为thought的维度
    def __init__(self, max_agents, thought_dim):
        super(Comm, self).__init__()

        self.hidden_dim = thought_dim // 2
        self.FC1 = nn.Linear(thought_dim, self.hidden_dim)
        self.BN1 = nn.LayerNorm(self.hidden_dim)
        # 此处hidden_dim为thought的1/2，使得输出正好为thought的维度
        self.bicnet = nn.GRU(input_size=self.hidden_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=1,
                             batch_first=False,
                             #nonlinearity='relu',
                             bidirectional=True)
        #self.FC2 = nn.Linear(thought_dim, thought_dim)
    '''
    thought_comb : n_agent, batch_size, thought_dim
    '''

    def forward(self, thought_comb):
        thought_comb = self.BN1(F.relu(self.FC1(thought_comb)))
        bi_output, _ = self.bicnet(thought_comb)
        #i_output = self.FC2(bi_output)
        return F.relu(bi_output)


class Critic(nn.Module):

    def __init__(self, dim_observation, dim_action):
        super(Critic, self).__init__()

        self.FC1 = nn.Linear(dim_observation + dim_action, 128)
        self.BN1 = nn.LayerNorm(128)
        self.FC2 = nn.Linear(128, 64)
        self.BN2 = nn.LayerNorm(64)
        self.FC3 = nn.Linear(64, 1)

    # obs: batch_size * obs_dim
    # obs_acts : batch_size * (obs_dim+act_dim)
    def forward(self, obs, acts):
        combined = th.cat([obs, acts], -1)
        result = self.BN1(F.relu(self.FC1(combined)))
        result = self.BN2(F.relu(self.FC2(result)))
        # return F.tanh(self.FC3(result))
        return self.FC3(result)


class Actor_I(nn.Module):
    def __init__(self, dim_observation, dim_thought):
        super(Actor_I, self).__init__()
        self.FC1 = nn.Linear(dim_observation, dim_thought)
        self.BN1 = nn.LayerNorm(dim_thought)
        self.FC2 = nn.Linear(dim_thought, dim_thought)
        #self.BN2 = nn.LayerNorm(dim_thought)

    def forward(self, obs):
        x = self.FC1(obs)
        x = F.relu(x)
        x = self.BN1(x)
        x = self.FC2(x)
        #x = self.BN2(x)
        # 此处是否应该采用relu之后的数值？
        return F.relu(x)


class Actor_II(nn.Module):
    def __init__(self, dim_thought, dim_action, discrete_action=False):
        super(Actor_II, self).__init__()
        self.FC1 = nn.Linear(dim_thought * 2, 128)
        self.BN1 = nn.LayerNorm(128)
        self.FC2 = nn.Linear(128, 64)
        self.BN2 = nn.LayerNorm(64)
        self.FC3 = nn.Linear(64, dim_action)
        if discrete_action:
            self.out_fn = lambda x: x
        else:
            self.out_fn = F.tanh

    def forward(self, thought_before, thought_after):
        combined = th.cat([thought_before, thought_after], -1)
        x = F.relu(self.FC1(combined))
        x = self.BN1(x)
        x = F.relu(self.FC2(x))
        x = self.BN2(x)
        # 此处是否应该采用relu之后的数值？
        # 此处应该用什么输出？
        return self.out_fn(self.FC3(x))

# 2020-12-13 delete sigmoid module to using exploration outside
class Attention(nn.Module):
    def __init__(self, dim_thought):
        super(Attention, self).__init__()
        self.FC1 = nn.Linear(dim_thought, 64)
        self.BN1 = nn.LayerNorm(64)
        self.FC2 = nn.Linear(64, 32)
        self.BN2 = nn.LayerNorm(32)
        self.FC3 = nn.Linear(32, 1)


    # modified to just linear network without activation fucntion especially remove relu function.
    def forward(self, thought):
        x = F.relu(self.FC1(thought))
        x = self.BN1(x)
        x = F.relu(self.FC2(x))
        x = self.BN2(x)
        #x = self.FC1(thought)
        #x = self.FC2(x)
        # 输出是否建立链路
        return F.sigmoid(self.FC3(x))


# The following is the DDPG models
class DDPG_Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(DDPG_Critic, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation
        act_dim = self.dim_action

        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024 + act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], -1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class DDPG_Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(DDPG_Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result