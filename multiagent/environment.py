import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents

        # set the type of agent
        if all([hasattr(a, 'adversary') for a in self.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                self.agents]
        else:
            self.agent_types = ['agent' for _ in self.agents]

        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = False
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            # 此处的obs_dim 被更改掉了，由于还返回了通信链路的信息。
            # 此行在有通信参与的时候运用
            obs_dim = len(observation_callback(agent, self.world)[0])
            # 此行在无通信参与的时候运用
            #obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n, non_atoc=1):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        step = self.world.step()
        # record observation for each agent
        for agent in self.agents:
            reward_n.append(self._get_reward(agent))
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        #reward = reward_n
        #if self.shared_reward:
            #reward_n = [reward] * self.n
        #########12-09##############
        #######################
        #for typical revise
        if non_atoc:
            reward = np.mean(reward_n)
            #if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n
    def get_info(self):
        info_n = []
        for agent in self.agents:
            info_n.append(self._get_info(agent))
        return info_n

    # get info used for benchmarking
    def _get_info(self, agent):
        #if self.info_callback is None:
        return {}
        #return self.info_callback(agent, self.world)

    def _get_evl_data(self):
        if self.info_callback is None:
            return {}
        return self.info_callback(self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            # 2021-02-07
            sensitivity = 1.0
            #sensitivity = 10.0
            #sensitivity = 15.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human', weights_prob=None):
        if weights_prob is not None:
            weights_prob = weights_prob.detach().squeeze().cpu().numpy()
            # weights_prob_sort = np.argsort(weights_prob)
            # weights_prob[weights_prob_sort[:-2]] = 0
        gap_fig = 26
        # time_record = [1,3,5,20]
        time_record = []
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(1000, 1000)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.landmarks:
                geom = rendering.make_circle(entity.size)
                # geom = rendering.Image(r'C:\Users\Alex_sheep\Desktop\user.png',25*2,25*2)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color, alpha=0.7)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)



        # Add the agents and their movement

        from multiagent import rendering
        #if self.time %gap_fig ==0 or self.time==1:
        # if self.time %gap_fig ==0 or self.time in time_record:
        if True:
            if hasattr(self, 'render_geoms_a') and hasattr(self, 'render_geoms_r'):
                for viewer in self.viewers:
                    for geom in self.render_geoms_a:
                        if geom in viewer.geoms:
                            viewer.geoms.remove(geom)
                    for geom in self.render_geoms_r:
                        if geom in viewer.geoms:
                            viewer.geoms.remove(geom)
                    for geom in self.render_geoms_r_c:
                        if geom in viewer.geoms:
                            viewer.geoms.remove(geom)

            self.render_geoms_a = []
            self.render_geoms_xform_a = []

            self.render_geoms_r = []
            self.render_geoms_xform_r = []

            self.render_geoms_r_c = []
            self.render_geoms_xform_r_c = []


        for entity in self.world.agents:
            #if self.time != 24:
            geom = rendering.make_circle(entity.size)
            if 'agent' in entity.name:
                # geom.set_color(*entity.color, alpha=0.5)
                geom.set_color(*entity.color, alpha=self.time / 25.0 *0.35+0.05)
            else:
                geom.set_color(*entity.color)
            #else:
            #    geom = rendering.Image(r'C:\Users\Alex_sheep\Desktop\uav_1.png', 40 * 2, 40 * 2)
            xform = rendering.Transform()
            geom.add_attr(xform)
            self.render_geoms_a.append(geom)
            self.render_geoms_xform_a.append(xform)

        for idx, entity in enumerate(self.world.agents):
            # if self.time != 24:
            if idx not in [0,2,4]:
                alpha = 0.6
            else:
                alpha =  0.6
            #     continue

            def angle(vector1, vector2):
                unit_v1 = vector1 / np.linalg.norm(vector1)
                unit_v2 = vector2 / np.linalg.norm(vector2)
                cos_ = np.dot(unit_v1, unit_v2)
                sin_ = np.cross(unit_v1, unit_v2)
                return np.arctan2(sin_, cos_)

            obs = self._get_obs(entity)[0]

            # obs_length = int(obs.shape[0] / 2)
            # sum_position = np.array([0.0,0.0])
            # for i in range(obs_length):
            #     cur_pos = obs[(i) * 2:(i + 1) * 2] /2
            #     cur_pos *= weights_prob[idx][i]
            #     sum_position += cur_pos
            # position = [np.array([0.0, 0.0]),sum_position,np.array([0.0, 0.0])]

            position = []
            obs_length = int(obs.shape[0] / 2)
            for i in range(obs_length):
                cur_pos = obs[(i) * 2:(i + 1) * 2]/5
                position.append(cur_pos)
            angle_lst = []
            for pos in position:
                angle_lst.append(angle(np.array([0.0, 1.0]), pos))
            angle_sort = np.argsort(angle_lst)
            position = [position[i] for i in angle_sort]

            position_draw = []
            for i,cur_pos in enumerate(position):
                cur_pos = cur_pos/ np.linalg.norm(cur_pos+0.00001)
                cur_pos *= weights_prob[idx][i]

                # get orthogonal
                ort_vec = np.array([1.0, - cur_pos[0]/cur_pos[1]])
                ort_vec = ort_vec/ np.linalg.norm(ort_vec+0.00001)

                position_draw.append(0.95*cur_pos-0.01*ort_vec)
                position_draw.append(cur_pos)
                position_draw.append(0.95*cur_pos+0.01*ort_vec)
                position_draw.append(np.array([0.0,0.0]))

            position_draw = [np.array([0.0,0.0])]+position_draw
            position_draw.append(position_draw[0])
            position_draw = np.array(position_draw)



            geom = rendering.make_polygon(position_draw,filled=True)

            if 'agent' in entity.name:
                geom.set_color(*entity.color, alpha=alpha)
                # geom.set_color(*entity.color, alpha=self.time / 25.0 * 0.35 + 0.05)
            else:
                geom.set_color(*entity.color)
            # else:
            #    geom = rendering.Image(r'C:\Users\Alex_sheep\Desktop\uav_1.png', 40 * 2, 40 * 2)
            xform = rendering.Transform()
            geom.add_attr(xform)
            self.render_geoms_r.append(geom)
            self.render_geoms_xform_r.append(xform)


            # 增加本地的长度
            geom = rendering.make_circle(entity.size*(1+2*weights_prob[idx][0]))
            geom.set_color(*entity.color, alpha=self.time / 25.0 * 0.35 + 0.05)
            xform = rendering.Transform()
            geom.add_attr(xform)
            self.render_geoms_r_c.append(geom)
            self.render_geoms_xform_r_c.append(xform)

        # add geoms to viewer
        for viewer in self.viewers:
            # viewer.geoms = []
            #if self.time % gap_fig != 0 and self.time !=1:
            # if self.time % gap_fig != 0 and self.time not in time_record:
            #     continue
            for geom in self.render_geoms_a:
                viewer.add_geom(geom)
            for geom in self.render_geoms_r:
                viewer.add_geom(geom)
            for geom in self.render_geoms_r_c:
                viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            #######################################################
            ########################here change the cam_range###############################
            #######################################################
            # ************************** #
            # cam_range = 1
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range,
                                       pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.landmarks):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                #print(*entity.state.p_pos)
            for e, entity in enumerate(self.world.agents):
                self.render_geoms_xform_a[e].set_translation(*entity.state.p_pos)
                # if e not in [0,2,4]:
                #      continue
                self.render_geoms_xform_r[e].set_translation(*entity.state.p_pos)
                self.render_geoms_xform_r_c[e].set_translation(*entity.state.p_pos)

                #print(*entity.state.p_pos)
            # render to display or array
            #results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))
            results.append(self.viewers[i].render(return_rgb_array = True))
        self.time += 1
        self.time = self.time % 25
        return results
    # render environment
    def _render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            #print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for i in range(len(self.world.landmarks)):
                #geom = rendering.Image(r'D:\GitCode\MultiRLCom\numbers\l_ '+str(i)+r'.PNG',0.05,0.05)
                geom = rendering.make_circle(self.world.landmarks[0].size)

                xform = rendering.Transform()
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

            self.render_geoms_a = []
            self.render_geoms_xform_a = []
            for i in range(len(self.world.agents)):
                #geom = rendering.Image(r'D:\GitCode\MultiRLCom\numbers\a_ '+str(i)+r'.PNG',0.05,0.05)
                geom = rendering.make_circle(self.world.agents[0].size)
                geom.set_color(*self.world.agents[0].color, alpha=0.3)
                xform = rendering.Transform()
                geom.add_attr(xform)
                self.render_geoms_a.append(geom)
                self.render_geoms_xform_a.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                # viewer.geoms = []
                for geom in self.render_geoms_a:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 50
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.landmarks):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            for e, entity in enumerate(self.world.agents):
                self.render_geoms_xform_a[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n



class MultiAgentEnvSimple(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = False
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = reward_n
        #if self.shared_reward:
            #reward_n = [reward] * self.n

        # get the relative distance between all agents
        dists = [[np.sqrt(np.sum(np.square(j.state.p_pos - l.state.p_pos))) for l in self.agents] for j in self.agents]

        return obs_n, reward_n, done_n, info_n, dists

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        dists = [[np.sqrt(np.sum(np.square(j.state.p_pos - l.state.p_pos))) for l in self.agents] for j in self.agents]
        return obs_n, dists

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 6.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(900,900)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx