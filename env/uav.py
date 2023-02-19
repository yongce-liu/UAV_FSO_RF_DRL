import gym
import numpy as np
from matplotlib import pyplot as plt

from env.arg_data import CarsPath
from env.channel import get_fso_gain, get_rf_gain, power_distribute
from env.store_file import Buffer

# car_speed = 15  # m/s
car_force = 5  # m/s^2
uav_height = 100  # m
# target_rate = 4.0e2  # Mbs
slot_time = 1  # s
rf_power = 10  # in dBm each
fso_power = 5  # in dBm each aver


class MakeEnv(gym.Env):
    def __init__(self, set_num, car_speed, target_rate):
        self.car_num = set_num
        self.car_speed = car_speed
        # load
        self.cars_path = CarsPath()
        self._max_episode_steps = self.cars_path.max_time
        # store
        self.buffer = Buffer(max_time=self._max_episode_steps + 1, car_num=self.car_num)
        self.p_rf_max = rf_power + 10 * np.log10(self.car_num)  # all power in dBm
        self.p_fso_max = fso_power * np.ones(shape=(self.car_num,))  # average power in dBm
        # edge constraint
        self.target_rate = target_rate
        self.delta_rate = target_rate * 1.0  # Mbps
        # [-500, -500, 0] -> [500, 500, 100]
        self.uav_acc_edge = np.array([0, 5], dtype=np.float32)  # m/s^2

        self.uav_velocity_edge = np.array([0, 10], dtype=np.float32)  # m/s
        
        self.env_edge = np.array([[-500, 500], [-500, 500], [0, 100]], dtype=np.float32)  # m
        # [0, 1], advice: 2 ** n
        observation_spaces = gym.spaces.Box(low=np.zeros(shape=(self.car_num + 2,), dtype=np.float32),
                                            high=np.ones(shape=(self.car_num + 2,), dtype=np.float32))
        self.observation_space = observation_spaces
        # [-1, 1]
        action_spaces = gym.spaces.Box(low=-1 * np.ones(shape=(2,), dtype=np.float32),
                                       high=np.ones(shape=(2,), dtype=np.float32))
        self.action_space = action_spaces

    def reset(self):
        self.time = 0  # slot = 1s, max_time = 1000s
        self.buffer.clear()  # reset Buffer
        temp_car_init_pos = self.cars_path.load(speed=self.car_speed, force=car_force, num=self.car_num)
        self.obj_point = self.cars_path.obj_pos
        # [-200, 200], uav-pos in m
        self.uav_pos = np.mean(temp_car_init_pos, axis=0) + np.array([0, 0., uav_height])
        # self.uav_pos = np.array([-450., 0, uav_height])
        # accelerate [-pi, pi], [0, 5]
        self.uav_acc_xy = np.zeros(shape=(2,), dtype=np.float32)
        self.pre_acc_xy = self.uav_acc_xy
        # velocity [0, pi]
        self.vel_theta = 0.
        self.uav_velocity_xy = np.zeros(shape=(2,), dtype=np.float32)
        # self.vel_mod = np.linalg.norm(self.uav_velocity_xy)

        state = self.deal_data()

        return state

    def step(self, action):
        self.time += 1
        info = {}
        if self.time < self._max_episode_steps:
            done = False
        else:
            done = True

        # 处理动作空间变换
        acc_theta = action[0] * np.pi  # [-pi, pi]
        # if np.abs(acc_theta) < np.pi*0.01:
        #     acc_theta *= 0.

        acc_mod = self.uav_acc_edge[1] * (action[1] + 1) / 2  # [-acc_edge, acc_edge]
        if acc_mod < self.uav_acc_edge[1] * 0.01:
            acc_mod *= 0.

        _theta = self.vel_theta + acc_theta  # [-pi, pi]

        if _theta <= -np.pi:
            acc_theta = (_theta + 2 * np.pi)
        elif _theta > np.pi:
            acc_theta = (_theta - 2 * np.pi)
        else:
            acc_theta = _theta
        # cal accelerate
        
        self.pre_acc_xy = self.uav_acc_xy
        self.uav_acc_xy = acc_mod * np.r_[np.cos(acc_theta), np.sin(acc_theta)]
        
        # constrain of accelarate
        # temp = 0.

        # uav position
        delta_tran = self.uav_velocity_xy * slot_time + \
                     0.5 * self.uav_acc_xy * (slot_time ** 2)
        
        # if np.linalg.norm(delta_tran) < self.env_edge[0, 1] * 0.01:
        #     delta_tran *= 0.
        
        self.uav_pos += np.r_[delta_tran, 0]
        # cal velocity
        self.uav_velocity_xy += self.uav_acc_xy * slot_time
        # 矫正
        self.rectify_pos()
        self.vel_mod = np.linalg.norm(self.uav_velocity_xy)
        # 计算夹角
        if self.uav_velocity_xy[1] > 0:
            self.vel_theta = np.arccos(self.uav_velocity_xy[0] / (self.vel_mod + 1e-10))
        else:
            self.vel_theta = -np.arccos(self.uav_velocity_xy[0] / (self.vel_mod + 1e-10))
        # whether position exceed
        # 超速
        if self.vel_mod > self.uav_velocity_edge[1]:
            ratio = self.uav_velocity_edge[1] / self.vel_mod
            self.uav_velocity_xy *= ratio
            # temp += (1 / ratio - 1) * 0.2
        # temp += (np.linalg.norm(self.delta_acc_acc_xy) - 1) * 0.3

        state = self.deal_data()
        reward = self.get_reward()

        return state, reward, done, info

    def render(self):
        now_time = self.time
        if not now_time % 10:
            plt.ion()  # 将画图模式改为交互模式
            plt.draw()
            # 设置三维图形模式
            ax = plt.axes(projection='3d')
            ax.set(xlabel='X', ylabel='Y', zlabel='Z',
                   title='Trans', xlim=self.env_edge[0],
                   ylim=self.env_edge[1], zlim=self.env_edge[2])
            # X = np.linspace(-R * 2, R * 2, 10)
            # Y = np.linspace(-R * 2, R * 2, 10)
            # X, Y = np.meshgrid(X, Y)
            # ax.plot_surface(X, Y, X * 0 + uav_height, alpha=0.2)
            # 线条
            for i in range(self.obj_point.shape[0]):
                left_point = self.obj_point[i, 0:3]
                right_point = self.obj_point[i, 3:]
                x = left_point[0]
                y = left_point[1]
                z = left_point[2]
                dx = right_point[0] - x
                dy = right_point[1] - y
                dz = right_point[2] - z
                
                xx = np.linspace(x, x+dx, 2)
                yy = np.linspace(y, y+dy, 2)
                zz = np.linspace(z, z+dz, 2)
                
                yy2, zz2 = np.meshgrid(yy, zz)
                ax.plot_surface(np.full_like(yy2, x), yy2, zz2, color='xkcd:light blue')
                ax.plot_surface(np.full_like(yy2, x+dx), yy2, zz2, color='xkcd:light blue')
                
                xx2, yy2 = np.meshgrid(xx, yy)
                ax.plot_surface(xx2, yy2, np.full_like(xx2, z), color='xkcd:light blue')
                ax.plot_surface(xx2, yy2, np.full_like(xx2, z+dz), color='xkcd:light blue')
                
                xx2, zz2= np.meshgrid(xx, zz)
                ax.plot_surface(xx2, np.full_like(yy2, y), zz2, color='xkcd:light blue')
                ax.plot_surface(xx2, np.full_like(yy2, y+dy), zz2, color='xkcd:light blue')

            ax.scatter3D(
                self.uav_pos[0], self.uav_pos[1], self.uav_pos[2], 'r*')
            # print(self.uav_pos[2])
            trans = self.buffer.uav_info['position'][:now_time]
            ax.plot3D(trans[:, 0], trans[:, 1], trans[:, 2], 'r')
            data = self.buffer.car_info
            for i in range(self.car_num):
                name = 'car_' + str(i)
                temp = data[name][:now_time]
                ax.plot3D(temp[:, 0], temp[:, 1], temp[:, 2], '--')
                # if self.h_fso[i] == 0:
                #     temp = self.cars_positions_xy[i][now_time, :]
                #     temp = np.c_[self.uav_position, np.r_[temp, 0]]
                #     ax.plot3D(temp[0], temp[1], temp[2])
                #     # print(temp)
                #     plt.pause(1)
            # plt.view
            plt.pause(0.1)
            plt.ioff()
            plt.clf()

    def seed(self, seed=None):
        seed = np.random.seed(seed)
        return [seed]

    def deal_data(self):
        self.delta_acc_acc_xy = self.uav_acc_xy - self.pre_acc_xy
        # print(delta_acc_acc_xy)
        self.pre_acc_xy = self.uav_acc_xy
        inter_index, self.cars_pos_list, self.distance = self.cars_path.get_inter_distance(time=self.time,
                                                                                           point=self.uav_pos)
        # if inter_index.any():
        #     print('jiac')
        self.h_fso = get_fso_gain(nlos_flag=inter_index, uav_pos=self.uav_pos, distance=self.distance, car_pos=np.array(self.cars_pos_list))
        self.h_rf = get_rf_gain(nlos_flag=inter_index, distance=self.distance)
        # print(self.h_fso, self.h_rf)

        self.r_rf, self.p_rf, self.r_fso, self.p_fso = \
            power_distribute(p_rf=self.p_rf_max, h_rf=self.h_rf, p_fso=self.p_fso_max, h_fso=self.h_fso,
                             target_rate=self.target_rate)

        self.r_all = self.r_rf + self.r_fso
        # print(self.r_all)

        self.store()

        state = np.clip(np.r_[(self.uav_velocity_xy / self.uav_velocity_edge[1] + 1) / 2,
                              (self.distance + np.random.normal(loc=0, scale=2, size=(self.car_num,))) * 2 ** -10,
                        ], 0, 1)
        # state = np.r_[(self.vel_theta / np.pi + 1) / 2,
        #             self.vel_mod / self.uav_velocity_edge[1],
        #             (self.distance + np.random.normal(loc=0, scale=1., size=(car_num,))) / 1000,
        #             np.abs((self.r_all - target_rate) / target_rate)
        #             ]
        # state = np.clip(np.r_[(self.vel_theta / np.pi + 1) / 2,
        #                       self.vel_mod / self.uav_velocity_edge[1],
        #                       self.r_all / 80
        #                       ], 0, 1)

        return state.astype(np.float32)

    def get_reward(self):
        # temp = np.abs((self.r_all - self.delta_rate) / self.delta_rate)
        # x1 = np.exp(-1.5 * temp.max()) - 1.
        # delta_acc = np.clip(np.linalg.norm(self.uav_acc_xy - self.pre_acc_xy) - 1, 0, np.inf)
        # x2 = np.exp(-1.3 * delta_acc) - 1.
        # reward = x1 * 1 + x2 * 0.
        x1 = (self.r_all - self.delta_rate)/self.delta_rate
        x2 = np.linalg.norm(self.uav_acc_xy - self.pre_acc_xy)/self.uav_acc_edge[1]

        reward = np.exp(-1.5 * np.abs(x1).max()) - 1.

        return reward

    def store(self):
        uav = [self.uav_pos, self.uav_velocity_xy, self.uav_acc_xy]
        car = self.cars_pos_list
        rate = [self.r_fso, self.r_rf, self.r_all, self.r_all.mean()]
        channel = [self.h_fso, self.p_fso, self.h_rf, self.p_rf]
        self.buffer.update(uav_info=uav, car_info=car, rate_info=rate, channel_info=channel)

    def rectify_pos(self):
        # [x, y, h]
        reward = 0
        coeff = np.array([1, -1])
        delta_x = (self.uav_pos[0] - self.env_edge[0]) * coeff
        delta_y = (self.uav_pos[1] - self.env_edge[1]) * coeff
        # 矫正 x
        index1 = delta_x < 0
        if index1.any():
            self.uav_pos[0] = self.env_edge[0][index1] * .9
            self.uav_velocity_xy[0] *= -0.1
            reward = -0.5
        # 矫正 y
        index2 = (delta_y < 0)
        if index2.any():
            self.uav_pos[1] = self.env_edge[1][index2] * .9
            self.uav_velocity_xy[1] *= -0.1
            reward = -0.5

        # return reward

    def numpy_cube_one(x=0, y=0, z=0, dx=50, dy=50, dz=50):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        xx = np.linspace(x, x+dx, 2)
        yy = np.linspace(y, y+dy, 2)
        zz = np.linspace(z, z+dz, 2)
        xx2, yy2 = np.meshgrid(xx, yy)
        ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
        ax.plot_surface(xx2, yy2, np.full_like(xx2, z+dz))
        yy2, zz2 = np.meshgrid(yy, zz)
        ax.plot_surface(np.full_like(yy2, x), yy2, zz2)
        ax.plot_surface(np.full_like(yy2, x+dx), yy2, zz2)
        xx2, zz2= np.meshgrid(xx, zz)
        ax.plot_surface(xx2, np.full_like(yy2, y), zz2)
        ax.plot_surface(xx2, np.full_like(yy2, y+dy), zz2)
        #坐标及其刻度隐藏
        plt.title("Cube")
        plt.show()

    @property
    def max_episode_steps(self):
        return self._max_episode_steps
