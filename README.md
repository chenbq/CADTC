## Pytorch implementation of "Learning Effective Cooperation for Fresh Data Collection in UAV-Aided IoT Networks"

This is the github repo for the work "Learning Effective Cooperation for Fresh Data Collection in UAV-Aided IoT Networks".

###  Detailed settings regarding the neural networks and the training process for the paper are provided as follows.

The critic network and the policy module in actor network are implemented by the fully-connected neural network (FNN) with three hidden layers having [128, 128, 64] neurons. For the actor network, the encode module utilizes FNN with two layers having [128, 128] neurons, and the information extraction module uses bi-directional Gated Recurrent Unit network with one hidden layer having 64 neurons. For all networks, $\it{ReLU}$ is used as the activation function of the hidden layers. The output layer's activation function of the actor network is $\it{tanh}$. The number of episodes is $T = 25$, and $\delta = 1$ s. The hyper-parameter $\lambda=1$. The capacity of the replay buffer is $10^5$ and the batch size is $K=1024$, while the discount factor is $\gamma = 0.95$. We use the Adam optimizer with an initial learning rate of $0.001$. For the soft update of target networks,  we set $\tau = 0.001$.



###  Run the code

1. To train the model. run main.py
   The hyper-parameters can be set as in /configs/xxx.txt

### Technical Details
https://github.com/chenbq/CADTC/blob/main/Details%20instructions%20for%20CADTC.pdf
