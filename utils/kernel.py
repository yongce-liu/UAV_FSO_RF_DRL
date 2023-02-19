import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.normalization import Normalization, RewardScaling
from utils.ppo import PPO_continuous
from utils.replaybuffer import ReplayBuffer
from env.uav import MakeEnv




def evaluate_policy(args, env, agent, state_norm):
    times = 2 ** 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            # We use the deterministic policy during the evaluating
            a = agent.evaluate(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, seed, speed, target_rate, ROOT_PATH=None, load_path=None, s_mean_std=None):
    # 创建训练环境和评价环境
    env = MakeEnv(set_num=args.car_num, car_speed=speed, target_rate=target_rate)
    # When evaluating the policy, we need to rebuild an environment
    env_evaluate = MakeEnv(set_num=args.car_num, car_speed=speed, target_rate=target_rate)

    # 设置环境随机种子
    env.seed(seed)
    env_evaluate.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 获得环境相关参数
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    # Maximum number of steps per episode
    args.max_episode_steps = env.max_episode_steps
    args.max_train_steps = args.max_episode_steps * args.max_train_episodes
    # print("state_dim={}".format(args.state_dim))
    # print("action_dim={}".format(args.action_dim))
    # # 动作空间范围为对称 [-r, r]
    # print("max_action={}".format(args.max_action))
    # print("max_episode_steps={}".format(args.max_episode_steps))

    if not load_path:
        print(ROOT_PATH)
        if not os.path.exists(ROOT_PATH):
            os.makedirs(ROOT_PATH)
        # 写入当前训练参数
        with open(ROOT_PATH + '/setting.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in args.__dict__.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
        f.close()
        

        # 记录评价回合
        evaluate_num = 0  # Record the number of evaluations
        all_evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0
        episode_num = 0  # Record the num of episode during the training
        state_norm_info = {"mean":[], 'std':[]}


        # 存储状态、动作、奖励等
        replay_buffer = ReplayBuffer(args)
        agent = PPO_continuous(args)

        # Build a tensorboard
        # writer = SummaryWriter(log_dir=ROOT_PATH + '/runs/uav_fso_{}_seed_{}'.format(args.policy_dist, seed))
        writer = SummaryWriter(log_dir=ROOT_PATH + '/runs')
        # Trick 2:state normalization
        state_norm = Normalization(shape=args.state_dim)
        if args.use_reward_norm:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif args.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        while episode_num < args.max_train_episodes:
            # 初始化 单次 episode
            s = env.reset()
            if args.use_state_norm:
                s = state_norm(s)
            if args.use_reward_scaling:
                reward_scaling.reset()
            # episode_steps = 0
            ep_r = 0
            done = False
            while not done:
                # episode_steps += 1
                # Action and the corresponding log probability
                a, a_logprob = agent.choose_action(s)
                # 根据分布输出得到真正动作范围
                if args.policy_dist == "Beta":
                    # [0,1] -> [-max,max]
                    action = 2 * (a - 0.5) * args.max_action
                else:
                    action = a

                s_, r, done, _ = env.step(action)
                ep_r += r

                if args.use_state_norm:
                    s_ = state_norm(s_)

                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)
                '''
                When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                dw means dead or win,there is no next state s';
                but when reaching the max_episode_steps,there is a next state s' actually.
                but for this situation uav need to deal within a period of time
                it always has the next step, we only need it fly during a period of time.
                if done and episode_steps != args.max_episode_steps:
                '''
                if done:
                    dw = True
                else:
                    dw = False

                # Take the 'action'，but store the original 'a'（especially for Beta）
                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                total_steps += 1

                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer.count == args.batch_size:
                    agent.update(replay_buffer, total_steps, writer)
                    replay_buffer.count = 0
                    state_norm_info["mean"].append(state_norm.running_ms.mean)
                    state_norm_info["std"].append(state_norm.running_ms.std)

            writer.add_scalar('train/reward_ep', ep_r, global_step=total_steps)
            episode_num = episode_num + 1
            # Evaluate the policy every 'evaluate_freq' steps
            if episode_num % args.evaluate_episode_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                all_evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t train_episodes:{} \t".format(evaluate_num, evaluate_reward, episode_num))
                writer.add_scalar('evaluate/reward_ep', evaluate_reward, global_step=total_steps)
                
                # Save the best rewards
                if (evaluate_reward >= np.mean(all_evaluate_rewards[-5:])) and (evaluate_num >= 5):
                    path = ROOT_PATH + '/data_train'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    np.save(path+'/uav_fso_{}_num_{}_seed_{}_rewards{}.npy'.format(evaluate_num, args.policy_dist, seed, evaluate_reward), np.array(all_evaluate_rewards))
                    env_evaluate.buffer.save(path=ROOT_PATH+'/flydata/', episode=episode_num, target_rate=target_rate)
                    agent.save_policy(reward=evaluate_reward, path=ROOT_PATH + '/model/', episode_num=episode_num)
        
        if args.use_state_norm:
            pd.DataFrame(state_norm_info).to_csv(ROOT_PATH + '/norm.csv')
        writer.close()

    else:
        agent = PPO_continuous(args)
        agent.load_policy(name=load_path)
        rew_all = []
        for episode_num in range(1):
            s = env.reset()
            done = False
            ep_r = 0.
            while not done:
                if s_mean_std:
                    s = (s - s_mean_std[0]) / s_mean_std[1]
                # Action and the corresponding log probability
                a, a_logprob = agent.choose_action(s)
                # 根据分布输出得到真正动作范围
                if args.policy_dist == "Beta":
                    # [0,1]->[-max,max]
                    action = 2 * (a - 0.5) * args.max_action
                else:
                    action = a
                # env.render()
                next_state, reward, done, info = env.step(action)
                ep_r += reward
                s = next_state
            rew_all.append(ep_r)
            env.buffer.save(path='./', episode=2, target_rate=target_rate)
        print(rew_all)
        print(np.mean(rew_all))


if __name__ == '__main__':
    env_steps= 600
    parser = argparse.ArgumentParser("Hyper-parameters Setting for PPO-continuous")
    parser.add_argument("--max_train_episodes", type=int, default=8000, help="Maximum number of training steps")
    parser.add_argument("--evaluate_episode_freq", type=int, default=8 * 5, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=env_steps*8, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=env_steps, help="Minibatch size")

    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=2e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=4e-4, help="Learning rate of critic")

    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.98, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.25, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=2 ** 3, help="PPO parameter")

    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")

    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")

    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")

    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")

    parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    main(args, seed=1)



    