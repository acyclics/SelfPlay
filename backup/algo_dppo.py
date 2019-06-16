import sys
sys.path.append("~/Desktop/Mods/ai/projects/selfplay")
import tensorflow as tf
import argparse
import numpy as np
import gym
import os
import scipy.signal
import datetime
from time import time
from gym import wrappers
from tensorflow.python.training.summary_io import SummaryWriterCache
from utils import RunningStats, discount, add_histogram

class DPPO(object):
    def __init__(self, worker_id, state_dim, action_dim, action_lowest, action_highest, trajectory_buffer_size, 
                 minibatch_size, epochs, sigma_floor, epsilon, decay_steps, end_learning_rate,
                 vf_coef, learning_rate, replicas_to_aggregate, n_replicas, L2_REG):
        self.worker_id = worker_id
        is_chief = worker_id == 0
        self.sigma_floor = sigma_floor
        self.L2_REG = L2_REG
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_mean = (action_highest - action_lowest) / 2.0
        self.actions = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="actions")
        self.state = tf.placeholder(tf.float32, shape=[None] + list(self.state_dim), name="state")
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name="advantage")
        self.reward = tf.placeholder(tf.float32, shape=[None, 1], name="reward")
        self.trajectory = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions, 
                                                              "reward": self.reward, "advantage": self.advantage})
        self.trajectory = self.trajectory.shuffle(buffer_size=trajectory_buffer_size)
        self.trajectory = self.trajectory.batch(batch_size=minibatch_size)
        self.trajectory = self.trajectory.cache()
        self.trajectory = self.trajectory.repeat(epochs)
        self.trjtry_iterator = self.trajectory.make_initializable_iterator()
        current_batch = self.trjtry_iterator.get_next()
        pi_old, pi_old_parameters = self._ff_actor_nn(current_batch["state"], "pi_old")
        pi, pi_parameters = self._ff_actor_nn(current_batch["state"], "pi")
        pi_eval, _ = self._ff_actor_nn(self.state, "pi", reuse=True)
        vf_old, vf_old_parameters = self._ff_critic_nn(current_batch["state"], "vf_old")
        self.vf, vf_parameters = self._ff_critic_nn(current_batch["state"], "vf")
        self.vf_eval, _ = self._ff_critic_nn(self.state, "vf", reuse=True)
        self.sample_action = tf.squeeze(pi_eval.sample(1), axis=0, name="sample_action")
        self.eval_action = pi_eval.mean()
        self.global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("loss"):
            clip_epsilon = tf.train.polynomial_decay(epsilon, self.global_step, decay_steps, end_learning_rate, power=0.0)
            with tf.variable_scope("policy"):
                llh_ratio = tf.maximum(pi.prob(current_batch["actions"]), 1e-6) / tf.maximum(pi_old.prob(current_batch["actions"]), 1e-6)
                llh_ratio = tf.clip_by_value(llh_ratio, 0, 10)
                surrogate_obj1 = current_batch["advantage"] * llh_ratio
                surrogate_obj2 = current_batch["advantage"] * tf.clip_by_value(llh_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                pi_loss = -tf.reduce_mean(tf.minimum(surrogate_obj1, surrogate_obj2))
                tf.summary.scalar(name="loss", tensor=pi_loss)
            with tf.variable_scope("value_function"):
                clipped_value_estimate = vf_old + tf.clip_by_value(self.vf - vf_old, -clip_epsilon, clip_epsilon)
                vf1_loss = tf.squared_difference(clipped_value_estimate, current_batch["reward"])
                vf2_loss = tf.squared_difference(self.vf, current_batch["reward"])
                vf_loss = tf.reduce_mean(tf.maximum(vf1_loss, vf2_loss)) * 0.5
                tf.summary.scalar(name="loss", tensor=vf_loss)
            loss = pi_loss + vf_coef * vf_loss
            tf.summary.scalar("total", loss)
        with tf.variable_scope("train"):
            self.global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=replicas_to_aggregate, total_num_replicas=n_replicas)
            self.train_op = optimizer.minimize(loss, global_step=self.global_step, var_list=pi_parameters + vf_parameters)
            self.sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        with tf.variable_scope("update_old"):
            self.update_pi_old_op = [oldp.assign(p) for p, oldp in zip(pi_parameters, pi_old_parameters)]
            self.update_vf_old_op = [oldp.assign(p) for p, oldp in zip(vf_parameters, vf_old_parameters)]
        tf.summary.scalar("value", tf.reduce_mean(self.vf))
        tf.summary.scalar("sigma", tf.reduce_mean(pi.stddev()))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def _ff_actor_nn(self, state, name, reuse=False):
        l2_weights_regu = tf.contrib.layers.l2_regularizer(self.L2_REG)
        with tf.variable_scope(name, reuse=reuse):
            layer1 = tf.layers.dense(inputs=state, units=400, activation=tf.nn.relu, kernel_regularizer=l2_weights_regu, name="pi_layer1")
            layer2 = tf.layers.dense(inputs=layer1, units=400, activation=tf.nn.relu, kernel_regularizer=l2_weights_regu, name="pi_layer2")
            mu = tf.layers.dense(inputs=layer2, units=self.action_dim, activation=tf.nn.tanh, kernel_regularizer=l2_weights_regu, name="pi_mu")
            log_sigma = tf.get_variable(name="pi_sigma", shape=self.action_dim, initializer=tf.zeros_initializer())
            dist = tf.distributions.Normal(loc=mu * self.action_mean, scale=tf.maximum(tf.exp(log_sigma), self.sigma_floor))
        parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, parameters

    def _ff_critic_nn(self, state, name, reuse=False):
        l2_weights_regu = tf.contrib.layers.l2_regularizer(self.L2_REG)
        with tf.variable_scope(name, reuse=reuse):
            layer1 = tf.layers.dense(inputs=state, units=400, activation=tf.nn.relu, kernel_regularizer=l2_weights_regu, name="vf_layer1")
            layer2 = tf.layers.dense(inputs=layer1, units=400, activation=tf.nn.relu, kernel_regularizer=l2_weights_regu, name="vf_layer2")
            vf = tf.layers.dense(inputs=layer2, units=1, kernel_regularizer=l2_weights_regu, name="vf_output")
        parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, parameters

    def update(self, s, a, r, adv, sess):
        start = time()
        episode_time = []
        sess.run([self.update_pi_old_op, self.update_vf_old_op, self.trjtry_iterator.initializer], 
                 feed_dict={self.state: s, self.actions: a, self.reward: r, self.advantage: adv})
        while True:
            try:
                episode_start = time()
                summary, step, _ = sess.run([self.summarise, self.global_step, self.train_op])
                episode_time.append(time() - episode_start)
            except tf.errors.OutOfRangeError:
                break
        print("Worker_%i Trained in %.3fs at %.3fs/batch. Global step %i" % (self.worker_id, time() - start, np.mean(episode_time), step))
        return summary
    
    def nn_predict(self, s, sess, stochastic=True):
        if stochastic:
            action, value = sess.run([self.sample_action, self.vf_eval], feed_dict={self.state: s[np.newaxis,:]})
        else:
            action, value = sess.run([self.eval_action, self.vf_eval], feed_dict={self.state: s[np.newaxis,:]})
        return action[0], np.squeeze(value)

def start_parameter_server(pid, spec):
    cluster = tf.train.ClusterSpec(spec)
    server = tf.train.Server(cluster, job_name="ps", task_index=pid)
    print("Starting PS #{}".format(pid))
    server.join()

class Worker(object):
    def __init__(self, ENVIRONMENT, SUMMARY_DIR, spec, EP_MAX, worker_id, state_dim, action_dim, action_lowest, action_highest, trajectory_buffer_size, 
                 minibatch_size, batch_size, epochs, sigma_floor, epsilon, decay_steps, end_learning_rate,
                 vf_coef, learning_rate, replicas_to_aggregate, n_replicas, L2_REG, GAMMA, LAMBDA):
        self.worker_id = worker_id
        self.SUMMARY_DIR = SUMMARY_DIR
        self.EP_MAX = EP_MAX
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA
        self.minibatch_size = minibatch_size
        self.batch_size = batch_size
        if ENVIRONMENT == "gimbal":
            self.env = gimbal
        elif ENVIRONMENT == "test":
            self.env = gym.make('Pendulum-v0')
            state_dim = self.env.observation_space.shape
            action_dim = self.env.action_space.shape[0]
            action_highest = self.env.action_space.high
            action_lowest = self.env.action_space.low
        else:
            print(ENVIRONMENT, "not found")
        print("Starting Worker #{}".format(self.worker_id))
        cluster = tf.train.ClusterSpec(spec)
        self.server = tf.train.Server(cluster, job_name="worker", task_index=self.worker_id)
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % self.worker_id, cluster=cluster)):
            if self.worker_id == 0:
                self.env = wrappers.Monitor(self.env, os.path.join(SUMMARY_DIR, ENVIRONMENT), video_callable=None)
            self.dppo = DPPO(worker_id=worker_id, state_dim=state_dim, action_dim=action_dim, action_lowest=action_lowest,
                             action_highest=action_highest, trajectory_buffer_size=trajectory_buffer_size, 
                             minibatch_size=minibatch_size, epochs=epochs, sigma_floor=sigma_floor, epsilon=epsilon,
                             decay_steps=decay_steps, end_learning_rate=end_learning_rate, vf_coef=vf_coef,
                             learning_rate=learning_rate, replicas_to_aggregate=replicas_to_aggregate, n_replicas=n_replicas, L2_REG=L2_REG)
    def work(self):
        hooks = [self.dppo.sync_replicas_hook]
        sess = tf.train.MonitoredTrainingSession(master=self.server.target, is_chief=(self.worker_id == 0),
                                                 checkpoint_dir=self.SUMMARY_DIR,
                                                 save_summaries_steps=None, save_summaries_secs=None, hooks=hooks)
        if self.worker_id == 0:
            writer = SummaryWriterCache.get(self.SUMMARY_DIR)
        t, episode, terminal = 0, 0, False
        buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
        rolling_r = RunningStats()
        while not sess.should_stop() and not (episode > self.EP_MAX and self.worker_id == 0):
            s = self.env.reset()
            ep_r, ep_t, ep_a = 0, 0, []
            while True:
                a, v = self.dppo.nn_predict(s, sess)
                if t == self.batch_size:
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
                    bs, ba, br, badv = np.reshape(buffer_s, (t,) + self.dppo.state_dim), np.vstack(buffer_a), \
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
                    print('Worker_%i' % self.worker_id,
                          '| Episode: %i' % episode, "| Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
                    if self.worker_id == 0:
                        worker_summary = tf.Summary()
                        worker_summary.value.add(tag="Reward", simple_value=ep_r)
                        actions = np.array(ep_a)
                        for a in range(self.dppo.action_dim):
                            add_histogram(writer, "Action/Dim" + str(a), actions[:, a], episode)
                        try:
                            writer.add_summary(graph_summary, episode)
                        except NameError:
                            pass
                        writer.add_summary(worker_summary, episode)
                        writer.flush()
                    episode += 1
                    break
        self.env.close()
        print("Worker_%i finished" % self.worker_id)

def main(_):
    pass

if __name__ == '__main__':
    ENVIRONMENT = 'test'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    PORT_OFFSET = 100
    OUTPUT_RESULTS_DIR = ".\\outputs"
    parser.add_argument('--job_name', action='store', dest='job_name', help='Either "ps" or "worker"')
    parser.add_argument('--task_index', action='store', dest='task_index', help='ID number of the job')
    parser.add_argument('--timestamp', action='store', dest='timestamp', help='Timestamp for output directory')
    parser.add_argument('--workers', action='store', dest='n_workers', help='Number of workers')
    parser.add_argument('--agg', action='store', dest='n_agg', help='Number of gradients to aggregate')
    parser.add_argument('--ps', action='store', dest='ps', help='Number of parameter servers')
    args = parser.parse_args()
    N_WORKER = int(args.n_workers)
    N_AGG = int(args.n_agg)
    PS = int(args.ps)
    TIMESTAMP = str(args.timestamp)
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "DPPO", ENVIRONMENT, TIMESTAMP)
    if PS == 0:
        spec = {"worker": ["localhost:" + str(2222 + PS + i + PORT_OFFSET) for i in range(N_WORKER)]}
    else:
        spec = {"ps": ["localhost:" + str(2222 + i + PORT_OFFSET) for i in range(PS)],
                "worker": ["localhost:" + str(2222 + PS + i + PORT_OFFSET) for i in range(N_WORKER)]}
    if args.job_name == "ps":
        tf.app.run(start_parameter_server(int(args.task_index), spec))
    elif args.job_name == "worker":
        w = Worker(ENVIRONMENT=ENVIRONMENT, SUMMARY_DIR=SUMMARY_DIR, spec=spec, EP_MAX=1000, worker_id=int(args.task_index),
                   state_dim=0, action_dim=0, action_lowest=0, action_highest=0, trajectory_buffer_size=10000, 
                   minibatch_size=32, batch_size=8192, epochs=10, sigma_floor=0.0, epsilon=0.1, decay_steps=1e5, end_learning_rate=0.01,
                   vf_coef=1.0, learning_rate=0.0001, replicas_to_aggregate=N_AGG, n_replicas=N_WORKER, L2_REG=0.001,
                   GAMMA=0.99, LAMBDA=0.95)
        tf.app.run(w.work())
