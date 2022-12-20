import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils.misc import categorical_sample

epsilon = 1e-6
LOG_SIG_MIN = -20
LOG_SIG_MAX = 2

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
            #self.out_fn = F.sigmoid
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class MLPOMNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, latent_dim=16, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPOMNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fcom1 = nn.Linear(input_dim, latent_dim)
        self.fcom2 = nn.Linear(latent_dim, latent_dim)
        self.fc1 = nn.Linear(input_dim+latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
            #self.out_fn = F.sigmoid
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h_0m = self.nonlin(self.fcom1(self.in_fn(X)))
        h_1m = self.nonlin(self.fcom2(h_0m))
        combined = torch.cat((self.in_fn(X), h_1m), dim=1)
        h1 = self.nonlin(self.fc1(combined))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class GaussianNet(nn.Module):

    def __init__(self,input_dim, out_dim, hidden_dim=64, action_scale = 1.0, action_bias = 0.0):
        super(GaussianNet, self).__init__()
        self.input_size = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(hidden_dim, out_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim, out_dim)  # logvariance layer

        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, x, training=True, ret_prob=False):
        out = self.fc1(x)
        out = self.relu(out)
        mean = self.fc21(out)
        log_std = self.fc22(out)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        if training:
            if ret_prob:
                return action, log_prob
            else:
                return action
        else:
            return mean


class DiscretNet(nn.Module):

    def __init__(self,input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu, norm_in=False, onehot_dim=0):
        super(DiscretNet, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, x, training=True, ret_prob=False):
        onehot = None
        if type(x) is tuple:
            x, onehot = x
        inp = self.in_fn(x)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)

        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        int_act, act = categorical_sample(probs, use_cuda=on_gpu)


        log_probs = F.log_softmax(out, dim=1).gather(1, int_act)

        if training:
            if ret_prob:
                return act, log_probs, (out**2).mean()
            else:
                return act
        else:
            return act