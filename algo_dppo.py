import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import gym
import scipy.signal
from time import time
from gym import wrappers
from tensorflow.python.training.summary_io import SummaryWriterCache
from utils import RunningStats, discount, add_histogram
from sp_utils import read_hyperparameters

class DPPO(object):
    def __init__(self, environment, wid, ENTROPY_BETA = 0.0, LR = 0.0001, MINIBATCH = 32, EPOCHS = 10, EPSILON = 0.1, VF_COEFF = 1.0,           
                 L2_REG = 0.001, SIGMA_FLOOR = 0.0):
        self.L2_REG = float(hyperparameters[15])
        self.SIGMA_FLOOR = float(hyperparameters[16])
        ENTROPY_BETA = float(hyperparameters[9])
        LR = float(hyperparameters[10])
        MINIBATCH = int(hyperparameters[11])
        EPOCHS = int(hyperparameters[12])
        EPSILON = float(hyperparameters[13])
        VF_COEFF = float(hyperparameters[14])
        L2_REG = float(hyperparameters[15])
        SIGMA_FLOOR = float(hyperparameters[16])
        BATCH_BUFFER_SIZE = int(hyperparameters[17])

        self.s_dim, self.a_dim = environment.observation_space.shape, environment.action_space.shape[0]
        self.a_bound = (environment.action_space.high - environment.action_space.low) / 2
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.wid = wid
        is_chief = wid == 0

        self.state = tf.placeholder(tf.float32, [None] + list(self.s_dim), 'state')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions,
                                                           "rewards": self.rewards, "advantage": self.advantage})
        self.dataset = self.dataset.shuffle(buffer_size=BATCH_BUFFER_SIZE)
        self.dataset = self.dataset.batch(MINIBATCH)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(EPOCHS)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()

        pi_old, pi_old_params = self._build_anet(batch["state"], 'oldpi')
        pi, pi_params = self._build_anet(batch["state"], 'pi')
        pi_eval, _ = self._build_anet(self.state, 'pi', reuse=True)

        vf_old, vf_old_params = self._build_cnet(batch["state"], "oldvf")
        self.v, vf_params = self._build_cnet(batch["state"], "vf")
        self.vf_eval, _ = self._build_cnet(self.state, 'vf', reuse=True)

        self.sample_op = tf.squeeze(pi_eval.sample(1), axis=0, name="sample_action")
        self.eval_action = pi_eval.mean()  # Used mode for discrete case. Mode should equal mean in continuous
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('loss'):
            epsilon_decay = tf.train.polynomial_decay(EPSILON, self.global_step, 1e5, 0.01, power=0.0)
            with tf.variable_scope('policy'):
                ratio = tf.maximum(pi.prob(batch["actions"]), 1e-6) / tf.maximum(pi_old.prob(batch["actions"]), 1e-6)
                ratio = tf.clip_by_value(ratio, 0, 10)
                surr1 = batch["advantage"] * ratio
                surr2 = batch["advantage"] * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
                loss_pi = -tf.reduce_mean(tf.minimum(surr1, surr2))
                tf.summary.scalar("loss", loss_pi)
            with tf.variable_scope('value_function'):
                clipped_value_estimate = vf_old + tf.clip_by_value(self.v - vf_old, -epsilon_decay, epsilon_decay)
                loss_vf1 = tf.squared_difference(clipped_value_estimate, batch["rewards"])
                loss_vf2 = tf.squared_difference(self.v, batch["rewards"])
                loss_vf = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5
                tf.summary.scalar("loss", loss_vf)
            with tf.variable_scope('entropy'):
                entropy = pi.entropy()
                pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)
            loss = loss_pi + loss_vf * VF_COEFF + pol_entpen
            tf.summary.scalar("total", loss)
        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()
            opt = tf.train.AdamOptimizer(LR)
            opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=N_AGG, total_num_replicas=N_WORKER)
            self.train_op = opt.minimize(loss, global_step=self.global_step, var_list=pi_params + vf_params)
            self.sync_replicas_hook = opt.make_session_run_hook(is_chief)
        with tf.variable_scope('update_old'):
            self.update_pi_old_op = [oldp.assign(p) for p, oldp in zip(pi_params, pi_old_params)]
            self.update_vf_old_op = [oldp.assign(p) for p, oldp in zip(vf_params, vf_old_params)]
        tf.summary.scalar("value", tf.reduce_mean(self.v))
        tf.summary.scalar("policy_entropy", tf.reduce_mean(entropy))
        tf.summary.scalar("sigma", tf.reduce_mean(pi.stddev()))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def _build_anet(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(self.L2_REG)
        with tf.variable_scope(name, reuse=reuse):
            layer_1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l1")
            layer_2 = tf.layers.dense(layer_1, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l2")
            mu = tf.layers.dense(layer_2, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg, name="pi_mu")
            log_sigma = tf.get_variable(name="pi_sigma", shape=self.a_dim, initializer=tf.zeros_initializer())
            dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.maximum(tf.exp(log_sigma), self.SIGMA_FLOOR))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, params

    def _build_cnet(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(self.L2_REG)
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l1")
            l2 = tf.layers.dense(l1, 400, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l2")
            vf = tf.layers.dense(l2, 1, kernel_regularizer=w_reg, name="vf_output")
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, params

    def update(self, s, a, r, adv, sess):
        start = time()
        e_time = []
        sess.run([self.update_pi_old_op, self.update_vf_old_op, self.iterator.initializer],
                 feed_dict={self.state: s, self.actions: a, self.rewards: r, self.advantage: adv})
        while True:
            try:
                e_start = time()
                summary, step, _ = sess.run([self.summarise, self.global_step, self.train_op])
                e_time.append(time() - e_start)
            except tf.errors.OutOfRangeError:
                break
        #print("Worker_%i Trained in %.3fs at %.3fs/batch. Global step %i" % (self.wid, time() - start, np.mean(e_time), step))
        return summary

    def evaluate_state(self, state, sess, stochastic=True):
        if stochastic:
            action, value = sess.run([self.sample_op, self.vf_eval], {self.state: state[np.newaxis, :]})
        else:
            action, value = sess.run([self.eval_action, self.vf_eval], {self.state: state[np.newaxis, :]})
        return action[0], np.squeeze(value)

''' FUNCTIONS TO FACILITATE TRAINING '''

def start_parameter_server(pid, spec):
    cluster = tf.train.ClusterSpec(spec)
    server = tf.train.Server(cluster, job_name="ps", task_index=pid)
    print("Starting PS #{}".format(pid))
    server.join()

def InitAssignFn(scaffold, sess):
    sess.run(init_assign_op, init_feed_dict)

''' END of FUNCTIONS TO FACILITATE TRAINING '''

class Worker(object):
    def __init__(self, wid, spec, EP_MAX = 30, GAMMA = 0.99, LAMBDA = 0.95, BATCH = 8192):
        # Early stopping
        self.BEST_REWARD = -float("inf")
        self.EARLYSTOP = False
        self.earlystop_r = 0
        # Hyperparameters
        self.EP_MAX = int(hyperparameters[5])
        self.GAMMA = float(hyperparameters[6])
        self.LAMBDA = float(hyperparameters[7])
        self.BATCH = int(hyperparameters[8])
        self.wid = wid
        self.env = gym.make(ENVIRONMENT)
        print("Starting Worker #{}".format(wid))
        cluster = tf.train.ClusterSpec(spec)
        self.server = tf.train.Server(cluster, job_name="worker", task_index=wid)
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % wid, cluster=cluster)):
            if self.wid == 0:
                self.env = wrappers.Monitor(self.env, os.path.join(SUMMARY_DIR, ENVIRONMENT), video_callable=None)
            self.dppo = DPPO(self.env, self.wid)

    def work(self):
        hooks = [self.dppo.sync_replicas_hook]
        sess = tf.train.MonitoredTrainingSession(master=self.server.target, is_chief=(self.wid == 0),
                                                 checkpoint_dir=SUMMARY_DIR,
                                                 scaffold=scaffold,
                                                 save_summaries_steps=None, save_summaries_secs=None, hooks=hooks)
        if self.wid == 0:
            writer = SummaryWriterCache.get(SUMMARY_DIR)
        t, episode, terminal = 0, 0, False
        buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
        rolling_r = RunningStats()
        while not sess.should_stop() and not (episode > self.EP_MAX and self.wid == 0) and not self.EARLYSTOP:
            s = self.env.reset()
            ep_r, ep_t, ep_a = 0, 0, []
            while True:
                a, v = self.dppo.evaluate_state(s, sess)
                if t == self.BATCH:
                    rewards = np.array(buffer_r)
                    rolling_r.update(rewards)
                    rewards = np.clip(rewards / rolling_r.std, -10, 10)
                    v_final = [v * (1 - terminal)] 
                    values = np.array(buffer_v + v_final)
                    terminals = np.array(buffer_terminal + [terminal])
                    delta = rewards + self.GAMMA * values[1:] * (1 - terminals[1:]) - values[:-1]
                    advantage = discount(delta, self.GAMMA * self.LAMBDA, terminals)
                    returns = advantage + np.array(buffer_v)
                    advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)
                    bs, ba, br, badv = np.reshape(buffer_s, (t,) + self.dppo.s_dim), np.vstack(buffer_a), \
                                       np.vstack(returns), np.vstack(advantage)
                    graph_summary = self.dppo.update(bs, ba, br, badv, sess)
                    buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
                    t = 0
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_v.append(v)
                buffer_terminal.append(terminal)
                ep_a.append(a)
                a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
                s, r, terminal, _ = self.env.step(a)
                buffer_r.append(r)
                ep_r += r
                ep_t += 1
                t += 1
                if terminal:
                    percentage = int(float(episode) * 100 / float(self.EP_MAX))
                    print("Percentage: {0:3d}% | Worker_{1} | Episode: {2} | Reward: {3} | Steps: {4}".format(percentage, self.wid, episode, ep_r, ep_t), end='\r')
                    if self.wid == 0:
                        worker_summary = tf.Summary()
                        worker_summary.value.add(tag="Reward", simple_value=ep_r)
                        actions = np.array(ep_a)
                        for a in range(self.dppo.a_dim):
                            add_histogram(writer, "Action/Dim" + str(a), actions[:, a], episode)
                        try:
                            writer.add_summary(graph_summary, episode)
                        except NameError:
                            pass
                        writer.add_summary(worker_summary, episode)
                        writer.flush()
                        # Early stopping
                        self.earlystop_r += ep_r
                        if (episode + 1) % 100 == 0:
                            self.earlyStopping()
                            self.earlystop_r = 0
                    episode += 1
                    break
        self.env.close()
        print("\n", end="")
        print("Worker_%i finished" % self.wid)

    def earlyStopping(self):
        if self.earlystop_r <= self.BEST_REWARD:
            self.EARLYSTOP = True
        else:
            self.BEST_REWARD = self.earlystop_r

def main(_):
    pass

if __name__ == '__main__':
    hyperparameters = read_hyperparameters()
    ENVIRONMENT = str(hyperparameters[0])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if type(tf.contrib) != type(tf): tf.contrib._warning = None
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', action='store', dest='job_name', help='Either "ps" or "worker"')
    parser.add_argument('--task_index', action='store', dest='task_index', help='ID number of the job')
    parser.add_argument('--sample', action='store', dest='sample', help='Directory to sample pre-train model from. None if no sampling')
    parser.add_argument('--timestamp', action='store', dest='timestamp', help='Timestamp for output directory')
    args = parser.parse_args()
    N_WORKER = int(hyperparameters[1])
    N_AGG = int(hyperparameters[1]) - int(hyperparameters[2])
    PS = int(hyperparameters[3])
    TIMESTAMP = str(args.timestamp)
    SAMPLE = str(args.sample)
    SUMMARY_DIR = ".\\players\\tmp\\"
    if SAMPLE != "None":
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        ckpt = tf.train.latest_checkpoint("C:\\Users\\impec\\Desktop\\Mods\\ai\\projects\\selfplay\\outputs\\pendulum\\DPPO\\Pendulum-v0\\2019_06_18_03_25_39")
        init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(ckpt, variables_to_restore)
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(), init_fn=InitAssignFn)
    else:
        scaffold = None
    if PS == 0:
        spec = {"worker": ["localhost:" + str(2222 + PS + i) for i in range(N_WORKER)]}
    else:
        spec = {"ps": ["localhost:" + str(2222 + i) for i in range(PS)],
                "worker": ["localhost:" + str(2222 + PS + i) for i in range(N_WORKER)]}
    if args.job_name == "ps":
        tf.app.run(start_parameter_server(int(args.task_index), spec))
    elif args.job_name == "worker":
        w = Worker(int(args.task_index), spec)
        tf.app.run(w.work())

'''
https://arxiv.org/pdf/1710.03748.pdf adjustments:
- multiple rollouts in parallel for each agent and have separate optimizers for each agent
- We collect a large amount of rollouts from the parallel workers and for each agent optimize 
    the objective with the collected batch on 4 GPUs
- Instead of estimating a truncated generalized advantage estimate (GAE) from a small number of steps 
    per rollout, as in Schulman et al. (2017); Heess et al. (2017), we estimate GAE from the full rollouts. 
    This is important as the competition reward is a sparse reward given at the termination of the episode
'''
