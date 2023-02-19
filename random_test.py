import matplotlib.pyplot as plt
import numpy as np

from env.uav import MakeEnv
import time


# car_num = [1, 2, 3, 4, 5, 6, 7, 8]
car_num = [8]
seed = 20205598
speed = 10
targe_rate = [550]
for rate in targe_rate:
    for item_num in car_num:
        env = MakeEnv(set_num=item_num, car_speed=speed, target_rate=rate)
        env.seed(seed)
        np.random.seed(seed)
        state = env.reset()
        done = False
        r = []
        while not done:
            # act = env.action_space.sample()
            # env.render()
            # time.sleep(200)
            # print(env.distance.mean())
            act = np.array([0, -1])
            pos_temp = np.array([np.r_[env.cars_path.cars_pos[i][env.time, :], 100] for i in range(item_num)])
            # pos_temp = pos_temp + np.c_[np.random.normal(loc=0, scale=2, size=(1, 2)), 0]
            env.uav_pos = pos_temp.mean(axis=0)
            next_state, reward, done, info = env.step(act)
            # print(env.r_fso)
            # print(env.h_fso)
            # print(state)
            r.append(reward)
            state = next_state

        env.buffer.save('./', episode='1', target_rate=rate)
        # fig = plt.figure()
        print(sum(r))
        # plt.plot(r)
        # plt.pause(50)
