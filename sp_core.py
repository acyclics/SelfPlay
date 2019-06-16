'''
https://arxiv.org/pdf/1710.03748.pdf
'''
import numpy as np
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines import DDPG, PPO2, GAIL

class History:
    def __init__(self, n_players, buffer_max):
        self.players = []
        self.n_players = n_players
        self.max = buffer_max
    def add(self, player, parameters):
        self.players[player]+= parameters
        self.truncate()
    def truncate(self):
        if (len(self.players[player]) >= self.max):
            self.players[player] = self.players[player][int(len(self.player1)*0.5):]
    def sample(self, player):
        no = np.random.uniform(low=0, high=int(len(self.players[player])*0.8))
        return self.players[no]

class SelfPlay:
    def __init__(self, n_agents, buffer_max, n_cpu, rollout_ts):
        self.n_agents = n_agents
        self.histories = History(buffer_max)
        self.n_cpu = n_cpu
        self.rollout_ts = rollout_ts
        #train once for param
    def play(self):
        player_env = self.get_env("player1")
        player_env = SubprocVecEnv([lambda: player_env for i in range(self.n_cpu)])
        model = PPO2(policy=MlpPolicy, env=player_env, gamma=0.99, n_steps=100, ent_coef=0.01, learning_rate=0.00025, 
                vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, 
                verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)
        model.learn(total_timesteps=self.rollout_ts, callback=None, seed=None, log_interval=1, tb_log_name='PPO2', reset_num_timesteps=True)
        self.histories.add("player1", model.get_parameters())
        player_env = self.get_env("player2")
        player_env = SubprocVecEnv([lambda: player_env for i in range(self.n_cpu)])
        model = PPO2(policy=MlpPolicy, env=player_env, gamma=0.99, n_steps=100, ent_coef=0.01, learning_rate=0.00025, 
                vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, 
                verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)
        model.learn(total_timesteps=self.rollout_ts, callback=None, seed=None, log_interval=1, tb_log_name='PPO2', reset_num_timesteps=True)
        self.histories.add("player2", model.get_parameters())
    def get_env(self, player):
        if player == "player1":
            player1 = model.load_parameters(self.histories.player1[-1])
            player2 = self.histories.sample("player2")
            env = self.env1(player1, player2)
            return env
        if player == "player2":
            player2 = model.load_parameters(self.histories.player2[-1])
            player1 = self.histories.sample("player1")
            env = self.env2(player2, player1)
            return env
