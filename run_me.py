from utils.kernel import main
from train_args import args_list

target_rate = 550 # Mbps

def func(args, path_num, speed):
    main(args=args, seed=20205598, speed=speed, target_rate=target_rate, ROOT_PATH='./output/' +
         'speed_' + str(speed) + '/' + str(path_num))


if __name__ == "__main__":
    import os
    import shutil
    import multiprocessing
    if os.path.exists('./output'):
        shutil.rmtree('./output')

    times = len(args_list)
    # times = 14
    speed = [10, 15]
    
    assert times % 6 == 0

    for sped in speed:
        for j in range(int(times/6)):
            record = []
            for i in range(j*6, (j+1)*6):
                process = multiprocessing.Process(
                    target=func, args=(args_list[i], i, sped))
                process.start()
                record.append(process)  # 把t1线程装到线程池里
            for process in record:
                process.join()
