import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from types import SimpleNamespace

# number of agents that an agent can observe in proximity.
N_neighbor = 2
A_neighbor = 2
T_max = 25

config = {
    'UAV_n':5,
    'UE_n':10,
    'range':50,
    'd_max':10,
    'noise': np.power(10, -100/10),
    'P_ref': 500*np.power(10, -37.6/10),# -37.6 reference at 1m loss,
    'alpha': -2,
    'height': 50, #m,
    'p_los_B': 0.35,
    'p_los_C': 5,
    'data_rate_requirement': 0.05,  # np.array([0.1, 0.2, 0.3, 0.4]),
    'lambda_ratio': 5,
    'P_blada_power': 0.012/8*1.225*0.05*0.79*np.power(400*0.5,3),# delta/8*rho*s*A*omega^3*R^3,
    'U_tip_speed_2': 200*200
    }
config = SimpleNamespace(**config)


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(config.UAV_n)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 3
        # add landmarks
        world.landmarks = [Landmark() for i in range(config.UE_n)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.size = 1.6
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_pos = np.random.uniform(-config.range, config.range, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            # agent 需要加入什么状态？
        # position list
        landmark_pos_lst = []
        for i, landmark in enumerate(world.landmarks):
            new_rand = np.random.uniform(-config.range, config.range, world.dim_p)
            if len(landmark_pos_lst)==0:
                landmark.state.p_pos =new_rand
                landmark_pos_lst.append(new_rand)
            else:
                dists = [np.sqrt(np.sum(np.square(new_rand - a))) for a in landmark_pos_lst]
                min_dist = min(dists)
                while min_dist<5:
                    new_rand = np.random.uniform(-config.range, config.range,  world.dim_p)
                    dists = [np.sqrt(np.sum(np.square(new_rand - a))) for a in landmark_pos_lst]
                    min_dist = min(dists)
                landmark.state.p_pos = new_rand
                landmark_pos_lst.append(new_rand)
            landmark.state.p_vel = np.zeros(world.dim_p)
            #增加landmark的计数，用于记录当前的AoI
            landmark.state.n_count = int(1)
            for l in world.landmarks: l.state.reward_calc_n = 0



    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        #print(agent1.state.p_pos)
        #print(agent2.state.p_pos)
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_out_range(self, agent):
        return True if np.max(np.abs(agent.state.p_pos)) > config.range else False

    def reward(self, agent, world):

        rew = .0
        # parameters rename
        noise = config.noise
        P_ref = config.P_ref
        alpha = config.alpha
        height = config.height
        p_los_B = config.p_los_B
        p_los_C = config.p_los_C
        R_0 = config.data_rate_requirement
        # Three steps:
        # 1. physical layer computation
        # coverage range computation to find which sensors to be served
        # beta_0 channel gain for 1 meter, M data size, B bandwidth, \tau is dt, sigma is noise, h is height.
        # R = sqrt(\beta_0 * P / (2^{M/B/tau}-1)/ sigma^2 - h^2  )

        # just consider LOS channel
        coverage_r = np.sqrt(  P_ref/ (np.power(2.0, R_0 )-1)/ np.square(noise) - np.square(height) )
        coverage_r = np.minimum(coverage_r, config.d_max)

        rew = 0

        acc_mat = np.zeros([config.UE_n, config.UAV_n])
        dis_mat = np.array([[np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents] for l in
                            world.landmarks])
        dis_mat_idx_descend = dis_mat.argsort(1)  # for given UE the order of UAVs.
        for idx_l, l in enumerate(world.landmarks):
            if dis_mat[idx_l, dis_mat_idx_descend[idx_l, 0]] <= coverage_r:
                acc_mat[idx_l, dis_mat_idx_descend[idx_l, 0]] = 1.0

        # 2. AoI update
        # update the state of sensors (landmarks)
        n_access_UAV = acc_mat.sum(1)
        n_access_UE = acc_mat.sum(0)  # number of users access to a given UAV
        # sum_data_rate_lst = []


        #lambda_AoI = 0 0.2 0.4 0.6 0.8 1.0
        lambda_AoI = self.lambda_AoI
        # print(np.where(n_access_UAV>0))
        # record for evaluation
        AoI_list = np.zeros(len(world.landmarks))
        for idx_l, l in enumerate(world.landmarks):


            if l.state.reward_calc_n == 0:
                #l.state.n_count += 1
                if n_access_UAV[idx_l] > 0:
                    # find the serving UAV index
                    idx_a = np.where(acc_mat[idx_l, :] == 1.0)[0]
                    # record the sev
                    l.state.n_count =  int(1)
                else:
                    l.state.n_count += int(1)
            l.state.reward_calc_n += 1
            if l.state.reward_calc_n == config.UAV_n:
                l.state.reward_calc_n = 0

            #rew -= lambda_AoI* l.state.n_count/T_max/len(world.landmarks)
            rew -= lambda_AoI*l.state.n_count/len(world.landmarks)
            AoI_list[idx_l] = l.state.n_count

        #print(np.where(np.array([l.state.n_count for l in world.landmarks])<2))
        # print(np.array([l.state.n_count for l in world.landmarks]))
        # 3. Energy update
        # compute the enrgy consumption for the UAV.
        lambda_power =  1-lambda_AoI
        P_blada_power = 0.012 / 8 * 1.225 * 0.05 * 0.79 * np.power(400 * 0.5, 3)  # delta/8*rho*s*A*omega^3*R^3
        U_tip_speed_2 = 200 * 200
        velocity_list = np.array([np.linalg.norm([a.state.p_vel[0], a.state.p_vel[0]]) for a in world.agents])
        power_consumption = P_blada_power * (1 + 3.0 * np.power(velocity_list, 2) / U_tip_speed_2)
        power_consumption_min = P_blada_power * (1 + 3.0 * np.power(0, 2) / U_tip_speed_2)
        power_consumption_max = P_blada_power * (1 + 3.0 * np.power(world.agents[0].max_speed*1.414, 2) / U_tip_speed_2)
        # power_consumption = power_consumption / (1 + P_blada_power*(1+3.0*np.power(world.agents[0].max_speed,2)/U_tip_speed_2))
        power_consumption = (power_consumption - power_consumption_min) / (
                    power_consumption_max - power_consumption_min)

        rew -= lambda_power*10*np.mean(power_consumption)
        world.aoi_list = AoI_list
        world.power_consumption = power_consumption
        world.velocity_list = velocity_list


        # team reward, so the penalty for each agent is the same.
        if agent.collide:
            for a in world.agents:
                for b in world.agents:
                    if a is b: continue
                    if self.is_collision(a, b):
                        rew -= 1/2

        for a in world.agents:
            if self.is_out_range(a):
                rew -= 10

        return rew

    def eval_data(self,world):
        #return (np.sum(world.data_rate_list), np.sum(world.true_data_list), world.trans_power,world.velocity_list)
        return (world.aoi_list, world.power_consumption, world.velocity_list)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        # for landmarks i.e., sensors
        entity_pos = []
        entity_count = []
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        idx_nearby = np.argsort(dists)
        idx_nearby = idx_nearby[:N_neighbor+1]
        landmarks_nearby = [world.landmarks[i] for i in idx_nearby]
        #landmarks_nearby = world.landmarks[idx_nearby]
        for entity in landmarks_nearby:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_count.append(float(entity.state.n_count) / T_max)

        # for agents i.e., UAVs
        #comm = []
        other_pos = []
        other_vel = []
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.agents]
        idx_nearby = np.argsort(dists)
        idx_nearby = idx_nearby[:A_neighbor+1]
        agents_nearby = [world.agents[i] for i in idx_nearby]
        for other in agents_nearby:
            if other is agent: continue
            #comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)

        return (np.array( np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + entity_pos).tolist() + entity_count),idx_nearby)

    def observation_(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_count = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_count.append(float(entity.state.n_count)/25.0)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return (np.array( np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + entity_pos).tolist() + entity_count),0)

# case test
if __name__ == '__main__':
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    import time
    scenario = scenarios.load('Data_collection' + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation)
    env.reset()
    for i in range(25):
        ation_n =[np.random.uniform(-1,1,2) for j in range(3)]
        obs_n, reward_n, done_n, info_n = env.step(ation_n)
        env.render()
        time.sleep(0.5)
        print([a.state.n_serv for a in world.landmarks])
        print(reward_n)
