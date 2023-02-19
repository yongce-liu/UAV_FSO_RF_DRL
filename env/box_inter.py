import numpy as np
import pandas as pd


class Objects:
    # 正面对物体 左下角顶点 右上角顶点
    def __init__(self, point):
        self.line_end = None
        self.line_start = None
        self.left_vertex = point[:, 0:3]
        self.right_vertex = point[:, 3:]
        self.obj_num = self.left_vertex.shape[0]

    def is_cross(self, line_start, line_end, line_length):
        ray_direction = line_end - line_start
        # 方向矢量
        ray_direction /= np.linalg.norm(ray_direction)
        is_cross_index = []
        for i in range(ray_direction.shape[0]):
            ray = ray_direction[i, :]
            start_point = line_start
            # ray_direction_norm = ray_direction/
            t_min = np.zeros(shape=(self.obj_num,), dtype=np.float32)
            # t_min = line_length[i] * np.ones(shape=(self.obj_num,), dtype=np.float32) * -1
            t_max = line_length[i] * np.ones(shape=(self.obj_num,), dtype=np.float32)
            for j in range(3):  # [x, y, z]
                if np.abs(ray[j]) < 1e-6:
                    if ((start_point[j] < self.left_vertex[:, j]) | (start_point[j] > self.right_vertex[:, j])).all():
                        is_cross_index.append(False)
                        break
                else:
                    inv_d = 1. / ray[j]
                    t1 = inv_d * (self.left_vertex[:, j] - start_point[j])
                    t2 = inv_d * (self.right_vertex[:, j] - start_point[j])
                    temp = np.c_[t1, t2]
                    t1 = temp.min(axis=1)
                    t2 = temp.max(axis=1)
                    t_min[t1 >= t_min] = t1[t1 >= t_min]
                    t_max[t2 <= t_max] = t2[t2 <= t_max]
                    if (t_min > t_max).all():
                        is_cross_index.append(False)
                        break
            if len(is_cross_index) == i:
                is_cross_index.append(True)

        return np.array(is_cross_index, dtype=bool)


if __name__ == "__main__":
    path = './data/speed_10_force_50/objects_data_Train10th.csv'
    data = pd.read_csv(path, index_col=0)
    print(data)
    # point1 = data.values
    point1 = np.array([[0, 1, 0, 1, 0, 1]])
    line = np.array([[0.5, 4, 4], [0.5, 1.2, 0]])
    # ponin = np.array([[-1, -1, -1, 1, 1, 1.]])
    print(line)
    obj = Objects(point1)
    a = line[0]
    print(a)
    b = line[1:]
    print(b)
    print(a - b)
    distance = np.linalg.norm((a - b), axis=1)
    print(distance)
    index = obj.is_cross(a, b, distance)
    print(index)
