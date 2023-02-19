import numpy as np
import pandas as pd

from env.box_inter import Objects

trace_indices = np.arange(start=1, stop=11, step=1, dtype=np.int32)
car_indices = np.arange(start=1, stop=9, step=1, dtype=np.int32)
cars_height = 2  # m
all_time = 600


# 坐标中心为[0, 0], x:[-500, 500], y:[500, 500]
class CarsPath:
    def __init__(self):
        self.cars_pos = None
        self.cars_num = None
        self.max_time = all_time

    def load(self, speed: int, force: int, num: int) -> np.ndarray:
        """
        :param speed: 车辆速度, [5, 10]
        :param force: 车辆的力, [5]
        :param num: 车辆数量, 1-8
        :return: 初始坐标
        """
        self.cars_num = num
        traces_name = np.random.choice(trace_indices, replace=False)
        cars_name = np.random.choice(car_indices, size=(self.cars_num,), replace=False)
        pre_path = "./env/data/speed_" + str(speed) + "_force_" + str(force) + \
                   "/trans_data_Train" + str(traces_name) + "th_User"
        self.cars_pos = []
        for i in cars_name:
            path = pre_path + str(i) + "th.csv"
            data = pd.read_csv(path, usecols=[0, 1]).values
            # 坐标变换
            data = data - 500.
            self.cars_pos.append(data)

        obj_path = "./env/data/speed_" + str(speed) + "_force_" + str(force) + \
                   "/objects_data_Train" + str(traces_name) + "th.csv"
        self.obj_pos = pd.read_csv(obj_path, index_col=0).values
        # 坐标变换
        self.obj_pos[:, 0:2] -= 500.
        self.obj_pos[:, 3:5] -= 500.
        # self.obj_pos[:, -1] += 5.

        self.box_inter = Objects(self.obj_pos)  # building
        self.cars_pos_arr = np.array([np.r_[self.cars_pos[i][0, :], cars_height] for i in range(self.cars_num)], dtype=np.float32)

        return self.cars_pos_arr

    def get_inter_distance(self, time: int, point: np.ndarray):
        """
        :param time: 找到时间
        :param point: 无人机坐标 (3,)
        :return: 是否交叉, 汽车坐标, 无人机与各车辆之间的距离 [m]
        """
        temp_cars_pos_list = [np.r_[self.cars_pos[i][time, :], cars_height] for i in range(self.cars_num)]
        # print(temp_cars_pos_list)
        temp_cars_pos = np.array(temp_cars_pos_list, dtype=np.float32)
        pos_vec = temp_cars_pos - point
        distance = np.linalg.norm(pos_vec, axis=1)
        # print(distance)
        # is inter
        inter_index = self.box_inter.is_cross(line_start=point,
                                              line_end=temp_cars_pos,
                                              line_length=distance)

        return inter_index, temp_cars_pos_list, distance


if __name__ == "__main__":
    a = CarsPath()
    a.load(5, 5, 2)
    # plt.figure()
    # for i in range(6):
    #     path = a.cars_pos[i]
    #     plt.plot(path[:, 0], path[:, 1])
    # plt.pause(10)
    print(a.get_inter_distance(a.max_time, np.array([20, 10, 100])))
