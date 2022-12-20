"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np


# For the wrapper of environment
class DummyVecEnv():

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]

        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.actions = None

    def step(self,actions):
        self.actions = actions
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):        
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def render(self, mode='human', weights_prob=None):
        results = []
        for env in self.envs:
            results.append(env.render(weights_prob = weights_prob))
        return results

    def _render(self):
        results = []
        for env in self.envs:
            results.append(env._render())
        return results

    def get_info(self):
        x=[]
        for e in range(self.num_envs):
            x.append(self.envs[e]._get_evl_data())
        return x

    def close(self):
        for env in self.envs:
            env.close()

    def get_world(self):
        for env in self.envs:
            env.world
        return env.world