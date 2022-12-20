import argparse
import torch
import time
import os
import numpy as np
import pickle
from types import SimpleNamespace
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import SuperReplayBuffer, ReplayBuffer, ReplayBufferATOC, ReplayBufferAttention, ReplayBufferSched, ReplayBufferFAM
from utils.env_wrappers import DummyVecEnv
from utils.misc import plot_curve_with_label
from algorithms import REGISTRY as alg_REGISTRY
from utils import REGISTRY as buf_REGISTRY
from tqdm import tqdm
import shutil

USE_CUDA = True# torch.cuda.is_available()
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled=False
continue_train = False # 是否从当前训练结果，继续训练（注意buff没有保存，必须从新获取）
episode_num = 0
max_links = 3 # ATOC最多建立的链路数
good_agent_len = 3 #prey 的数目

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action, lambda_AoI):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, lambda_AoI, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        #环境相关的参数：
        '''
        num_envs
        observation_space
        action_space
        agent_types
        '''
        return DummyVecEnv([get_env_fn(0)])
    else:
        print('multi-thread-has-not-been-surpportted')
        return

def transf_params(params):
    save_params = dict()
    for var_name in params:
        if len(var_name)==0:
            continue
        #if 'optim' in var_name:
        save_params.update({var_name:params[var_name]})
        '''else:
            temp_dict = {}
            for in_name in params[var_name]:
                temp_dict.update({in_name:params[var_name][in_name].shape})
            save_params.update({var_name:temp_dict})'''
    return save_params

def get_model_agent(config, env):
    ReplayBufferx = ReplayBuffer
    if config.agent_alg in ['ATOC','ATOC_sim','Sched','FAM']:
        ReplayBufferx = buf_REGISTRY[config.agent_alg]
    ReplayBufferx = SuperReplayBuffer
    replay_buffer_attention = ReplayBufferAttention(config.buffer_length, config.dim_thought)
    AgentNetClass = alg_REGISTRY[config.agent_alg]
    if continue_train == 1:
        # provide the model path to continue training
        model_path = Path('./models') / config.env_id / config.model_name / 'run5' / 'model.pt'
        AgentNet = AgentNetClass.init_from_save(model_path)
    else:

        arg_dict = {'dim_thought': config.dim_thought,
                    'gamma': config.gamma,
                    'tau':config.tau,
                    'lr':config.lr,
                    'hidden_dim':config.hidden_dim,
                    'max_agents':max_links,
                    'cuda' : USE_CUDA}
        arg_dict_maddpg = {'agent_alg':config.agent_alg,
                            'adversary_alg':config.adversary_alg,
                            'tau':config.tau,
                            'lr':config.lr,
                            'hidden_dim':config.hidden_dim,
                           'cuda' : USE_CUDA}
        if config.agent_alg in ['maddpg', 'ddpg', 'maacg','ommaddpg','cddpg']:
            AgentNet = AgentNetClass.init_from_env(env, **arg_dict_maddpg)
        else:
            AgentNet = AgentNetClass.init_from_env(env, **arg_dict)
            AgentNet.provide_env(env)


    return AgentNet, ReplayBufferx,replay_buffer_attention

def save_code(run_dir, config):
    code_dir = run_dir / 'codes'
    src_dir = Path('.')
    os.makedirs(code_dir)

    # file list
    file_list = ['main_all.py', 'main_eval.py', 'main_bench_data.py', 'plot_rewards.py']
    # dir list
    dir_list = ['algorithms', 'nets', 'configs', 'multiagent', 'utils']
    for file_i in file_list:
        shutil.copy2(src_dir /file_i, code_dir)  # complete target filename given

    for dir_i in dir_list:
        shutil.copytree(src_dir / dir_i, code_dir / dir_i)

def initial(config):
    # 生成模型文件夹和logger记录学习过程
    model_dir = Path('./models') / config.env_id / (config.model_name+'_middle')
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run

    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    # saving current code for logger
    save_code(run_dir, config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # save the lambda setting for multi-objective optimization
    with open('{}/'.format(run_dir)+str(config.lambda_AoI)+'.txt', 'w') as f:
        f.write(',' + str(config.lambda_AoI) + '\n')

    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    path_temp = '{}/all_reward.csv'.format(log_dir)
    mode_temp = 'w'

    return run_dir, log_dir, logger, path_temp, mode_temp


# observation pre-process
def pre_obs(obs_raw, nagents, agent_alg, n_rollout_threads):
    obs = np.array([[obs_raw[:, :, 0][0, i] for i in range(nagents)]])
    nearby_agents =None
    nearby_agents_mat = None
    if agent_alg in ['ATOC', 'ATOC_sim','FAM']:
        nearby_agents = np.array([[obs_raw[:, :, 1][0, i][:max_links] for i in range(nagents)]])

    if agent_alg == 'Sched':
        nearby_agents = [[obs_raw[:, :, 1][0, i][:max_links + 1].tolist() for i in range(nagents)]]
        for i in range(nagents):
            nearby_agents[0][i].remove(i)
        nearby_agents = np.array(nearby_agents)

    if nearby_agents is not None:
        nearby_agents_mat = np.zeros((nagents, n_rollout_threads, nagents))
        for i in range(n_rollout_threads):
            for j in range(nagents):
                nearby_agents_mat[j, i, nearby_agents[i, j, :]] = 1
    if agent_alg == 'FAM':
        nearby_agents = nearby_agents.swapaxes(0, 1)
    return obs, nearby_agents, nearby_agents_mat

def run(config):


    run_dir, log_dir, logger, path_temp, mode_temp = initial(config)

    # make environment
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action, config.lambda_AoI)

    # generate model and buffer
    AgentNet, ReplayBufferx,replay_buffer_attention = get_model_agent(config, env)



    # 初始化buffer
    # 此处通过nagents，避免了返回或者需要更多的obs输入，因为good agent不纳入其中。
    # replay_buffer = ReplayBufferx(config.buffer_length, AgentNet.nagents,
    #                              [obsp.shape[0] for obsp in env.observation_space],
    #                              [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
    #                               for acsp in env.action_space])

    replay_buffer = ReplayBufferx(config.buffer_length)
    t = 0 # for multiple thread environment

    # 初始化noise
    AgentNet.scale_noise(config.init_noise_scale)
    AgentNet.reset_noise()



    for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):

        episode_num = ep_i/5000.0
        AgentNet.update_episode_num(episode_num)
        # Debug Info
        if ep_i % 10000 == 0:
            print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))

        obs_raw = env.reset()

        # 对观测值进行处理
        obs, nearby_agents, nearby_agents_mat= pre_obs(obs_raw, AgentNet.nagents,
                                                       config.agent_alg, config.n_rollout_threads)

        AgentNet.prep_rollouts()

        # Adapting noise scale
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        AgentNet.scale_noise(
            config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        AgentNet.reset_noise()

        d_Q_list = []
        thought_list = []
        com_flag = False
        for et_i in range(config.episode_length):
            #print(et_i)
            # obs转换到torch的格式
            if config.agent_alg in ['maddpg','ddpg','maacg','ommaddpg','cddpg']:
                # 由于maddpg要使用enumerate因此，torch_obs以list形式组织，而不是torch
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False)
                                  for i in range(AgentNet.nagents)]
                if USE_CUDA:
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])).to('cuda'), requires_grad=False)
                                 for i in range(AgentNet.nagents)]
            else:
                torch_obs = Variable(torch.Tensor([obs[:,agent_idx,:] for agent_idx in range(AgentNet.nagents)]), requires_grad=False)
                if USE_CUDA:
                    torch_obs = torch_obs.to('cuda')

            # 获取action，step时，模型已经将data转换到cpu了。
            if config.agent_alg in ['ATOC','ATOC_sim']:
                # 此处应该通过attention的变化来确定nearby agent的变化，不能周围的neighbor变了，依然和原来的neighbor通信
                if et_i % config.comm_T == 0:
                    torch_agent_actions, thought_array, d_Q_array, comm_top = AgentNet.step(torch_obs, nearby_agents_mat, explore=True,com_flag=com_flag)
                else:
                    torch_agent_actions, thought_array, d_Q_array, comm_top = AgentNet.step(torch_obs, nearby_agents_mat,com_flag=com_flag,
                                                                                            explore=True, comm_top_in=comm_top, com_mode='reuse')
                d_Q_list += d_Q_array
                if len(thought_array)>0 and np.array([thought_array[i].data.numpy() for i in range(len(thought_array))]).max()==0:
                    print('error')
                thought_list += thought_array
            elif config.agent_alg == 'Sched':
                torch_agent_actions = AgentNet.step(torch_obs, nearby_agents, nearby_agents_mat,  explore=True)
            elif config.agent_alg == 'FAM':
                torch_agent_actions = AgentNet.step(torch_obs, nearby_agents, explore=True)
            else:
                torch_agent_actions = AgentNet.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            if len(obs_raw[0, 0, :])==3:
                # 此处处理obs中包括的prey的信息
                actions = [[ac[i] for ac in agent_actions]+[-obs_raw[:, :, 2][0][0][idx] for idx in range(good_agent_len)] for i in range(config.n_rollout_threads)]
                #actions = [[np.concatenate((ac[i], -obs_raw[:, :, 2][0, idx][0])) for idx, ac in enumerate(agent_actions)] for i in range(config.n_rollout_threads)]
            obs_raw, rewards, dones, infos = env.step(actions)

            next_obs, next_nearby_agents, next_nearby_agents_mat\
                = pre_obs(obs_raw, AgentNet.nagents,config.agent_alg, config.n_rollout_threads)


            # Rendering
            # if ep_i%50==0:
            #     env._render()
            #     time.sleep(0.05)

            # data should be prepared before push into buffers with structure
            # [ [num_agents, dim_object], ....]
            # [ [num_agents, dim_object],

            if config.agent_alg in ['ATOC','ATOC_sim']:
                comm_top_push = comm_top.permute(1, 0 ,2).detach().numpy() if comm_top.device=='cpu' \
                    else comm_top.permute(1, 0 ,2).detach().cpu().numpy()
            agent_actions = np.array(agent_actions)
            agent_actions = np.swapaxes(agent_actions, 0, 1)
            dones = dones.astype(float)

            if config.agent_alg in ['ATOC','ATOC_sim']:
                '''
                obs (1,7,16)
                agent_action [7, (1,2)]
                rewards (1,7)
                next_obs (1,7,16)
                dones (1,7)
                comm_top (7,1,7) torch
                nearby_agents (1,7,3)
                '''
                replay_buffer.push_batch([obs, agent_actions, rewards, next_obs, dones, comm_top_push],batch_size=1)
                if et_i%config.comm_T==0:
                    nearby_agents = next_nearby_agents
                    nearby_agents_mat = next_nearby_agents_mat
            elif config.agent_alg == 'FAM':
                replay_buffer.push_batch([obs, agent_actions, rewards, next_obs, dones,
                                   np.transpose(nearby_agents, (1,0,2)), np.transpose(next_nearby_agents, (1,0,2))],batch_size=1)
                nearby_agents = next_nearby_agents
                nearby_agents_mat = next_nearby_agents_mat
            elif config.agent_alg=='Sched':
                replay_buffer.push_batch([obs, agent_actions, rewards, next_obs, dones,
                                   nearby_agents_mat,np.transpose(nearby_agents, (1,0,2)),next_nearby_agents_mat, np.transpose(next_nearby_agents, (1,0,2))],batch_size=1)
                nearby_agents = next_nearby_agents
                nearby_agents_mat = next_nearby_agents_mat
            else:
                #print(next_obs.shape)
                replay_buffer.push_batch([obs, agent_actions, rewards, next_obs, dones],batch_size=1)


            obs = next_obs
            t += config.n_rollout_threads


            # Training and Updating Models
            if len(replay_buffer) >= config.batch_size and t % config.steps_per_update == 0:

                AgentNet.prep_training()
                #for i_net in range(config.n_rollout_threads):
                # #重复采样n次，然后做n次训练。update
                if config.agent_alg in ['maddpg','ddpg','maacg','ommaddpg']:
                    for a_i in range(AgentNet.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        AgentNet.update(sample, a_i, logger=logger)
                elif config.agent_alg=='cddpg':
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=USE_CUDA)
                    AgentNet.update(sample, logger=logger)
                else:
                    sample = replay_buffer.sample(config.batch_size,
                                                    to_gpu=USE_CUDA)
                    AgentNet.update(sample, logger=logger, time_update = et_i/float(config.episode_length))
                #if ep_i % config.steps_per_update == 0:


                # train attention unit
                if config.agent_alg in ['ATOC','ATOC_sim'] and len(replay_buffer_attention) >= 10:
                    # replay buffer 全部取出来
                    sample_attention = replay_buffer_attention.sample(len(replay_buffer_attention), to_gpu=USE_CUDA)
                    at_loss = AgentNet.update_attention(sample_attention)
                    # logger.add_scalar('agent%i/at_loss' % 1, at_loss, ep_i)
                    # clear replay buffer
                    replay_buffer_attention = ReplayBufferAttention(config.buffer_length, config.dim_thought)

                AgentNet.prep_rollouts()


        # Evaluating methods
        if ep_i % 100 ==0 :
            #AgentNet.mov_all_models()
            ep_rews = evaluate_alg(env, AgentNet, config, explore=False)
            logger.add_scalar('agent%i/mean_step_rewards' % -1, np.mean(ep_rews), ep_i)
            print('mean_episode_reward',np.mean(ep_rews))
            if os.path.exists(path_temp):
                mode_temp = 'a'
            with open(path_temp, mode_temp) as f:
                f.write(',' + str(np.mean(ep_rews)) + '\n')



        # Save model
        if ep_i % config.save_interval < config.n_rollout_threads:
            AgentNet.save(run_dir / 'model.pt')
            if AgentNet.cuda:
                AgentNet.mov_all_models('cuda')


    AgentNet.save(run_dir / 'model.pt')


    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

def evaluate_alg(env, AgentNet, config, explore = False):
    Eval_episode = 2

    mean_reward_vs_time = np.zeros((Eval_episode, config.episode_length))
    for ep_i in range(Eval_episode):
        obs_raw = env.reset()

        obs, nearby_agents, nearby_agents_mat = pre_obs(obs_raw, AgentNet.nagents,
                                                        config.agent_alg, config.n_rollout_threads)
        for et_i in range(config.episode_length):

            # rearrange observations to be per agent, and convert to torch Variable
            if config.agent_alg in ['maddpg', 'ddpg', 'maacg','ommaddpg','cddpg']:
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                      requires_grad=False)
                             for i in range(AgentNet.nagents)]
                if USE_CUDA:
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])).to('cuda'), requires_grad=False)
                                 for i in range(AgentNet.nagents)]
            else:
                torch_obs = Variable(torch.Tensor([obs[:, agent_idx, :] for agent_idx in range(AgentNet.nagents)]),
                                     requires_grad=False)
                if USE_CUDA:
                    torch_obs = torch_obs.to('cuda')
            # get actions as torch Variables
            if config.agent_alg in ['ATOC','ATOC_sim']:
                torch_agent_actions, thought_array, d_Q_array, comm_top = AgentNet.step(torch_obs, nearby_agents_mat,
                                                                                        explore=explore)
            elif config.agent_alg == 'FAM':
                torch_agent_actions = AgentNet.step(torch_obs, nearby_agents, explore=True)
            else:
                torch_agent_actions = AgentNet.step(torch_obs, explore=explore)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            obs_raw, rewards, dones, infos = env.step(actions)

            next_obs, next_nearby_agents, next_nearby_agents_mat \
                = pre_obs(obs_raw, AgentNet.nagents, config.agent_alg, config.n_rollout_threads)

            obs = next_obs
            if config.agent_alg not in ['ATOC','ATOC_sim'] or et_i % config.comm_T == 0:
                nearby_agents = next_nearby_agents
                nearby_agents_mat = next_nearby_agents_mat

            mean_reward_vs_time[ep_i, et_i] = np.mean(rewards)

    return np.mean(mean_reward_vs_time)


if __name__ == '__main__':
    #是否端点继续训练
    continue_flag = 0
    model_dir = Path('./configs')
    config_dir = model_dir/ 'ddpg_config.txt'
    with open(config_dir, 'r', encoding='utf-8') as f:
        #config = f.read()
        a = eval(f.read())
        config = SimpleNamespace(**a)
    lambda_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i_lambda in lambda_list:
        config.lambda_AoI = i_lambda
        run(config)