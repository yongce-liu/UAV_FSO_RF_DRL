import argparse


def my_args(params):
    parser = argparse.ArgumentParser(
        "Hyper-parameters Setting for PPO-continuous")

    parser.add_argument("--max_train_episodes", type=int,
                        default=params["max_train_episodes"], help="Maximum number of training steps")
    parser.add_argument("--evaluate_episode_freq", type=int,
                        default=params["evaluate_episode_freq"], help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--policy_dist", type=str,
                        default="Beta", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int,
                        default=params['batch_size'], help="Batch size")
    parser.add_argument("--mini_batch_size", type=int,
                        default=params['mini_batch_size'], help="Minibatch size")
    parser.add_argument("--car_num", type=int, default=8,
                        help="The number of cars.")

    parser.add_argument("--hidden_width", type=int,
                        default=params["hidden_width"], help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float,
                        default=params['lr_a'], help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float,
                        default=params['lr_c'], help="Learning rate of critic")

    parser.add_argument("--gamma", type=float,
                        default=params['gamma'], help="Discount factor")
    parser.add_argument("--lamda", type=float,
                        default=params['lamda'], help="GAE parameter")
    parser.add_argument("--epsilon", type=float,
                        default=params['epsilon'], help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int,
                        default=2 ** 3, help="PPO parameter")

    parser.add_argument("--use_adv_norm", type=bool,
                        default=params['use_adv_norm'], help="Trick 1:advantage normalization")

    parser.add_argument("--use_state_norm", type=bool,
                        default=params['use_state_norm'], help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool,
                        default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool,
                        default=False, help="Trick 4:reward scaling")

    parser.add_argument("--entropy_coef", type=float,
                        default=params['entropy_coef'], help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool,
                        default=params['use_lr_decay'], help="Trick 6:learning rate Decay")

    parser.add_argument("--use_grad_clip", type=bool,
                        default=params['use_grad_clip'], help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool,
                        default=False, help="Trick 8: orthogonal initialization")

    parser.add_argument("--set_adam_eps", type=float,
                        default=params['set_adam_eps'], help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float,
                        default=params['use_tanh'], help="Trick 10: tanh activation function")

    args = parser.parse_args()

    return args


env_steps = 600
# 6400 4M
# 12800 8M
# gamma 不要超过 0.99
# 尽量只使用 adv norm

arg_dict0 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 4e-4,
             "gamma": 0.98,
             "lamda": 0.98,
             "epsilon": 0.25,
             "use_adv_norm": True,
             "use_state_norm": False,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict1 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 4e-4,
             "gamma": 0.95,
             "lamda": 0.98,
             "epsilon": 0.25,
             "use_adv_norm": True,
             "use_state_norm": False,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict2 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 4e-4,
             "gamma": 0.90,
             "lamda": 0.98,
             "epsilon": 0.25,
             "use_adv_norm": True,
             "use_state_norm": False,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict3 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 4e-4,
             "gamma": 0.98,
             "lamda": 0.98,
             "epsilon": 0.25,
             "use_adv_norm": True,
             "use_state_norm": False,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict4 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 4e-4,
             "gamma": 0.98,
             "lamda": 0.96,
             "epsilon": 0.25,
             "use_adv_norm": True,
             "use_state_norm": False,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict5 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 2e-4,
             "gamma": 0.98,
             "lamda": 0.98,
             "epsilon": 0.25,
             "use_adv_norm": True,
             "use_state_norm": False,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict6 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 4e-4,
             "gamma": 0.98,
             "lamda": 0.98,
             "epsilon": 0.25,
             "use_adv_norm": True,
             "use_state_norm": False,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict7 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 1e-3,
             "gamma": 0.98,
             "lamda": 0.98,
             "epsilon": 0.25,
             "use_adv_norm": True,
             "use_state_norm": False,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict8 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 4e-4,
             "gamma": 0.98,
             "lamda": 0.98,
             "epsilon": 0.25,
             "use_adv_norm": True,
             "use_state_norm": False,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict9 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 4e-4,
             "gamma": 0.98,
             "lamda": 0.98,
             "epsilon": 0.25,
             "use_adv_norm": False,
             "use_state_norm": True,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict10 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 128,
             "lr_a": 2e-4,
             "lr_c": 4e-4,
             "gamma": 0.98,
             "lamda": 0.98,
             "epsilon": 0.25,
             "use_adv_norm": True,
             "use_state_norm": True,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}

arg_dict11 = {"max_train_episodes": int(6400),
             "evaluate_episode_freq": int(64),
             "batch_size": int(env_steps * 8),
             "mini_batch_size": int(env_steps / 2),
             "hidden_width": 256,
             "lr_a": 2e-4,
             "lr_c": 1e-3,
             "gamma": 0.97,
             "lamda": 0.95,
             "epsilon": 0.20,
             "use_adv_norm": True,
             "use_state_norm": True,
             "entropy_coef": 0.01,
             "use_lr_decay": True,
             "use_grad_clip": False,
             "set_adam_eps": False,
             "use_tanh": False}


temp_args_list = [arg_dict0, arg_dict1, arg_dict2, arg_dict3, arg_dict4, arg_dict5,
                  arg_dict6, arg_dict7, arg_dict8, arg_dict9, arg_dict10, arg_dict11]


args_list = []

for item in temp_args_list:
    args_list.append(my_args(item))
