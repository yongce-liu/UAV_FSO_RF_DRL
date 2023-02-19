import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Trick 8: orthogonal initialization


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        # Trick10: use tanh
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(
            torch.zeros(1, args.action_dim))  # We use 'nn.Paremeter' to train log_std automatically
        # Trick10: use tanh
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # [-1,1]->[-max_action,max_action]
        mean = self.max_action * torch.tanh(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        # To make 'log_std' have the same dimension as 'mean'
        log_std = self.log_std.expand_as(mean)
        # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        std = torch.exp(log_std)
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        # Trick10: use tanh
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_continuous():
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        # 策略网路
        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args).to(device)
        else:
            self.actor = Actor_Gaussian(args).to(device)
        # 评判网络
        self.critic = Critic(args).to(device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(
                self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(
                self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(
            s, dtype=torch.float, device=device), 0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().cpu().numpy().flatten()
        else:
            a = self.actor(s).detach().cpu().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(
            s, dtype=torch.float, device=device), 0)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                # The log probability density of the action
                a_logprob = dist.log_prob(a)
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action,
                                self.max_action)  # [-max,max]
                # The log probability density of the action
                a_logprob = dist.log_prob(a)
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def update(self, replay_buffer, total_steps, writer):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor(
            device=device)  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            # for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
            for delta, d in zip(reversed(deltas.flatten()), reversed(done.flatten())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float,
                               device=device).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(
                    1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1,
                                                                                             keepdim=True))  # shape(mini_batch_size X 1)

                # Only calculate the gradient of 'a_logprob_now' in ratios
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon,
                                    1 + self.epsilon) * adv[index]
                # Trick 5: policy entropy
                actor_loss = -torch.min(surr1, surr2) - \
                    self.entropy_coef * dist_entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            _lr_a, _lr_c = self.lr_decay(total_steps)

        writer.add_scalar('train/actor_loss', actor_loss.mean(), total_steps)
        writer.add_scalar('train/actor_lr', _lr_a, total_steps)
        writer.add_scalar('train/critic_loss', critic_loss.mean(), total_steps)
        writer.add_scalar('train/critic_lr', _lr_c, total_steps)
        writer.add_scalar('train/entropy_log', dist_entropy.mean(), total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
        return lr_a_now, lr_c_now

    def save_policy(self, reward, path, episode_num):
        if not os.path.exists(path + '/actor/'):
            os.makedirs(path + '/actor/')
        path_actor = path + '/actor/' + str(episode_num) + "-" + str(reward)
        torch.save(self.actor.state_dict(), path_actor)

        if not os.path.exists(path + '/critic/'):
            os.makedirs(path + '/critic/')
        path_critic = path + '/critic/' + str(episode_num) + "-" + str(reward)
        torch.save(self.critic.state_dict(), path_critic)

    def load_policy(self, name):
        self.actor.load_state_dict(torch.load(
            name, map_location=torch.device('cpu')))
