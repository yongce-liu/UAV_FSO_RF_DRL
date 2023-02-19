import os
from datetime import datetime

import numpy as np


class Buffer:
    def __init__(self, max_time: int, car_num: int):
        # self.save_path = "./output/fly_data/" + str(datetime.now())[0:10] + "-" + str(datetime.now())[11:13]
        # if not os.path.exists(self.save_path):
        #     os.mkdir(self.save_path)
        self.max_time = max_time
        self.time = 0
        self.car_num = car_num
        # # uav_pos-(2,), uav_acc-(2,), uav_vel-(2,)
        # self.uav_info = np.zeros(shape=(self.max_time, 3), dtype=object)
        # # car_pos-(2,) all-num
        # self.car_info = np.zeros(shape=(self.max_time, self.car_num), dtype=object)
        # # rate_fso-(num,), rate_rf-(num,), rate_all-(num,), rate_mean-(1,)
        # self.rate_info = np.zeros(shape=(self.max_time, self.car_num + 1), dtype=object)

    def update(self, uav_info: list, car_info: list, rate_info: list, channel_info: list):
        # uav
        self.uav_info['position'][self.time, :] = uav_info[0]
        self.uav_info['velocity'][self.time, :] = uav_info[1]
        self.uav_info['accelerate'][self.time, :] = uav_info[2]
        # car
        for i in range(self.car_num):
            name = "car_" + str(i)
            self.car_info[name][self.time, :] = car_info[i]
        # rate
        self.rate_info["fso_rate"][self.time, :] = rate_info[0]
        self.rate_info["rf_rate"][self.time, :] = rate_info[1]
        self.rate_info["all_rate"][self.time, :] = rate_info[2]
        self.rate_info["mean_rate"][self.time] = rate_info[3]
        # channel
        self.channel_info["gain_fso"][self.time, :] = channel_info[0]
        self.channel_info["power_fso"][self.time, :] = channel_info[1]
        self.channel_info["gain_rf"][self.time, :] = channel_info[2]
        self.channel_info["power_rf"][self.time, :] = channel_info[3]
        self.time += 1

    def save(self, path, episode: int, target_rate):
        if not os.path.exists(path):
            os.makedirs(path)
        # path = path + str(episode)
        np.save(path + "uav_" + str(target_rate), self.uav_info)
        np.save(path + "car_" + str(target_rate), self.car_info)
        np.save(path + "rate_" + str(target_rate), self.rate_info)

    def clear(self):
        # clear
        self.time = 0
        # uav_pos-(3,), uav_acc-(2,), uav_vel-(2,)
        self.uav_info = {"position": np.zeros(shape=(self.max_time, 3), dtype=np.float32),
                         "velocity": np.zeros(shape=(self.max_time, 2), dtype=np.float32),
                         "accelerate": np.zeros(shape=(self.max_time, 2), dtype=np.float32)}
        # car_pos-(2,) all-num
        self.car_info = {}
        for i in range(self.car_num):
            name = "car_" + str(i)
            temp = {name: np.zeros(shape=(self.max_time, 3), dtype=np.float32)}
            self.car_info.update(temp)
        # rate_fso-(num,), rate_rf-(num,), rate_all-(num,), rate_mean-(1,)
        self.rate_info = {"fso_rate": np.zeros(shape=(self.max_time, self.car_num), dtype=np.float32),
                          "rf_rate": np.zeros(shape=(self.max_time, self.car_num), dtype=np.float32),
                          "all_rate": np.zeros(shape=(self.max_time, self.car_num), dtype=np.float32),
                          "mean_rate": np.zeros(shape=(self.max_time,), dtype=np.float32)}
        self.channel_info = {"gain_fso": np.zeros(shape=(self.max_time, self.car_num), dtype=np.float32),
                             "power_fso": np.zeros(shape=(self.max_time, self.car_num), dtype=np.float32),
                             "gain_rf": np.zeros(shape=(self.max_time, self.car_num), dtype=np.float32),
                             "power_rf": np.zeros(shape=(self.max_time, self.car_num), dtype=np.float32)}


if __name__ == "__main__":
    memory = Buffer(max_time=10, car_num=2)
    memory.clear()
    uav_info_1 = np.array([1, 2, 3])
    uav_info_2 = np.array([2, 3])
    uav_info_3 = np.array([3, 4])
    uav_info = [uav_info_1, uav_info_2, uav_info_3]

    car_info_1 = np.array([1, 2, 4])
    car_info_2 = np.array([2, 3, 4])
    car_info = [car_info_1, car_info_2]

    rate_info_1 = np.array([1, 2])
    rate_info_2 = np.array([2, 3])
    rate_info_3 = np.array([3, 4])
    rate_info = [rate_info_1, rate_info_2, rate_info_3, rate_info_3.mean()]

    # rate_info_4 = rate_info_3.mean()

    print(rate_info)

    # rate_info_temp.append(rate_info_4)

    # print(rate_info_temp)

    memory.update(uav_info, car_info, rate_info)

    memory.save('train', '1')
