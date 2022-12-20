import numpy as np
import random
from torch import Tensor
from torch.autograd import Variable

class SuperReplayBuffer(object):
    def __init__(self, size):
        """Create Super Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def push(self, data):#obs_t, message, action, reward, obs_tp1, done):
        #data = (obs_t, message, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def push_batch(self, data, batch_size):

        for i in range(batch_size):
            data_i = []
            for data_entity in data:
                data_i.append(data_entity[i])
            self.push(data_i)


    def _encode_sample(self, idxes, to_gpu=False):
        #obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        l = len(self._storage[0])
        lists = [[] for _ in range(l)]
        for i in idxes:
            data = self._storage[i]
            #obs_t, message, action, reward, obs_tp1, done = data
            for j in range(l):
                lists[j].append(np.array(data[j],copy=False))
            #obses_t.append(np.array(obs_t, copy=False))
            #messages.append(np.array(message, copy=False))
            #actions.append(np.array(action, copy=False))
            #rewards.append(reward)
            #obses_tp1.append(np.array(obs_tp1, copy=False))
            #dones.append(done)
        for j in range(l):
            data_np = np.array(lists[j])
            data_swap = list(np.swapaxes(data_np, 0, 1))
            lists[j]  = [cast(x) for x in data_swap]
        return lists#np.array(obses_t), np.array(messages), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(int(batch_size))]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(int(batch_size))]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size, to_gpu=False):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes, to_gpu=to_gpu)

    def collect(self):
        return self.sample(-1)



class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim),np.float16))
            self.ac_buffs.append(np.zeros((max_steps, adim),np.float16))
            self.rew_buffs.append(np.zeros(max_steps,np.float16))
            self.next_obs_buffs.append(np.zeros((max_steps, odim),np.float16))
            self.done_buffs.append(np.zeros(max_steps))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]



class ReplayBufferATOC(object):
    """
    Replay Buffer for ATOC with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.comm_buffs = [] # 11-22 加入了comm_buff，用于表示当前的agent是否通信
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim),np.float16))
            self.ac_buffs.append(np.zeros((max_steps, adim),np.float16))
            self.rew_buffs.append(np.zeros(max_steps,np.float16))
            self.next_obs_buffs.append(np.zeros((max_steps, odim),np.float16))
            self.done_buffs.append(np.zeros(max_steps))
            self.comm_buffs.append(np.zeros((max_steps, num_agents),np.uint8))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    # comms为array，且维度为dim_agents, dim_agents，也就是comm_matrix
    def push(self, observations, actions, rewards, next_observations, dones, comms):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
                self.comm_buffs[agent_i] = np.roll(self.comm_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
            # 此处的特殊性，由于comms与当前其他部分保持一致
            self.comm_buffs[agent_i][self.curr_i:self.curr_i + nentries] = comms[agent_i, :]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.comm_buffs[i][inds]) for i in range(self.num_agents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]


class ReplayBufferSched(object):
    """
    Replay Buffer for ATOC with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, num_neaby=3):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.comm_buffs = [] # 11-22 加入了comm_buff，用于表示当前的agent是否通信
        self.agent_nearby = []
        self.next_comm_buffs = []
        self.next_agent_nearby = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim),np.float16))
            self.ac_buffs.append(np.zeros((max_steps, adim),np.float16))
            self.rew_buffs.append(np.zeros(max_steps,np.float16))
            self.next_obs_buffs.append(np.zeros((max_steps, odim),np.float16))
            self.done_buffs.append(np.zeros(max_steps))
            self.comm_buffs.append(np.zeros((max_steps, num_agents),np.uint8))
            self.agent_nearby.append(np.zeros((max_steps, num_neaby),np.uint8))
            self.next_comm_buffs.append(np.zeros((max_steps, num_agents),np.uint8))
            self.next_agent_nearby.append(np.zeros((max_steps, num_neaby),np.uint8))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    # comms为array，且维度为dim_agents, dim_agents，也就是comm_matrix
    def push(self, observations, actions, rewards, next_observations, dones, comms, nearby, n_comms, n_nearby):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
                self.comm_buffs[agent_i] = np.roll(self.comm_buffs[agent_i],
                                                   rollover)
                self.agent_nearby[agent_i] = np.roll(self.agent_nearby[agent_i],
                                                   rollover)
                self.next_comm_buffs[agent_i] = np.roll(self.next_comm_buffs[agent_i],
                                                   rollover)
                self.next_agent_nearby[agent_i] = np.roll(self.next_agent_nearby[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
            # 此处的特殊性，由于comms与当前其他部分保持一致
            self.comm_buffs[agent_i][self.curr_i:self.curr_i + nentries] = comms[agent_i, :]
            self.agent_nearby[agent_i][self.curr_i:self.curr_i + nentries] = nearby[agent_i, :]
            self.next_comm_buffs[agent_i][self.curr_i:self.curr_i + nentries] = n_comms[agent_i, :]
            self.next_agent_nearby[agent_i][self.curr_i:self.curr_i + nentries] = n_nearby[agent_i, :]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.comm_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.agent_nearby[i][inds]) for i in range(self.num_agents)],
                [cast(self.next_comm_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.next_agent_nearby[i][inds]) for i in range(self.num_agents)]
                )

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]

class ReplayBufferFAM(object):
    """
    Replay Buffer for AFM with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, num_neaby=3):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.agent_nearby = []
        self.next_agent_nearby = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim),np.float16))
            self.ac_buffs.append(np.zeros((max_steps, adim),np.float16))
            self.rew_buffs.append(np.zeros(max_steps,np.float16))
            self.next_obs_buffs.append(np.zeros((max_steps, odim),np.float16))
            self.done_buffs.append(np.zeros(max_steps))
            self.agent_nearby.append(np.zeros((max_steps, num_neaby),np.uint8))
            self.next_agent_nearby.append(np.zeros((max_steps, num_neaby),np.uint8))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    # comms为array，且维度为dim_agents, dim_agents，也就是comm_matrix
    def push(self, observations, actions, rewards, next_observations, dones, comms, nearby, n_comms, n_nearby):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
                self.agent_nearby[agent_i] = np.roll(self.agent_nearby[agent_i],
                                                   rollover)
                self.next_agent_nearby[agent_i] = np.roll(self.next_agent_nearby[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
            self.agent_nearby[agent_i][self.curr_i:self.curr_i + nentries] = nearby[agent_i, :]
            self.next_agent_nearby[agent_i][self.curr_i:self.curr_i + nentries] = n_nearby[agent_i, :]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.agent_nearby[i][inds]) for i in range(self.num_agents)],
                [cast(self.next_agent_nearby[i][inds]) for i in range(self.num_agents)]
                )

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]

class ReplayBufferAttention(object):
    """
    Replay Buffer of Attention Training for ATOC with parallel rollouts
    """
    def __init__(self, max_steps, num_thought):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_thought (int): Number of dimension for thought
        """
        self.max_steps = max_steps
        self.num_thought = num_thought
        self.thought_buffs = np.zeros((max_steps, num_thought),dtype=np.float16)
        self.d_Q_buffs = np.zeros((max_steps, 1),dtype=np.float16)

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    # thought，且维度为num_epside, dim_thought
    def push(self, thought, d_Q):
        nentries = thought.shape[0]
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.thought_buffs= np.roll(self.thought_buffs,
                                                   rollover)
            self.d_Q_buffs= np.roll(self.d_Q_buffs,
                                               rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps

        self.thought_buffs[self.curr_i:self.curr_i + nentries] = thought
        self.d_Q_buffs[self.curr_i:self.curr_i + nentries, 0] = d_Q
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        return ( cast(self.thought_buffs[inds]), cast(self.d_Q_buffs[inds]) )

'''
以下为unit test
'''

if __name__ == '__main__':

    test_d_Q = SuperReplayBuffer(100)
    for i in range(1000):
        observations = np.array(np.random.randn(1,3,24))
        actions = np.array(np.random.randn(1,3,2))
        rewards = np.array(np.random.randn(1,3,1 ))
        next_observations = np.array(np.random.randn(1,3,24))
        dones = np.array(np.random.randn(1,3, 1 ))
        comms = np.array(np.random.randn(1,3, 3))
        nearby = np.array(np.random.randn(1,3, 3))
        n_comms = np.array(np.random.randn(1,3, 3))
        n_nearby = np.array(np.random.randn(1,3, 3))
        test_d_Q.push_batch([observations, actions, rewards, next_observations, dones, comms, nearby, n_comms, n_nearby],batch_size=1)
        print(test_d_Q._next_idx)
    test_d_Q.push([observations, actions, rewards, next_observations, dones, comms, nearby, n_comms, n_nearby])
    print(test_d_Q.sample(5))
    '''for i in range(1000):
        attention = np.array(np.random.randn(26,128))
        d_Q = np.array(np.random.randn(26))
        test_d_Q.push(attention,d_Q)
        print(test_d_Q.curr_i)
    test_d_Q.push(attention, d_Q)
    print(test_d_Q.sample(5))'''
