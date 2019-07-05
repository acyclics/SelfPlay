import os
import tensorflow as tf
import numpy as np
from sp_utils import read_hyperparameters
import argparse
from gym import spaces
import sys

class Infer(object):
    def __init__(self, env_high, env_low, a_dim, s_dim):
        hyperparameters = read_hyperparameters()
        self.L2_REG = float(hyperparameters[15])
        self.SIGMA_FLOOR = float(hyperparameters[16])
        self.a_bound = (env_high - env_low) / 2
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.state = tf.placeholder(tf.float32, [None] + list(self.s_dim), 'state')
        self.pi_eval = self._build_anet(self.state, 'pi')
        self.eval_action = self.pi_eval.mean()

    def _build_anet(self, state_in, name):
        with tf.variable_scope(name):
            w_reg = tf.contrib.layers.l2_regularizer(self.L2_REG)
            layer_1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l1")
            layer_2 = tf.layers.dense(layer_1, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l2")
            mu = tf.layers.dense(layer_2, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg, name="pi_mu")
            log_sigma = tf.get_variable(name="pi_sigma", shape=self.a_dim, initializer=tf.zeros_initializer())
            dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.maximum(tf.exp(log_sigma), self.SIGMA_FLOOR))
        return dist

    def evaluate_state(self, state, sess):
        action = sess.run(self.eval_action, {self.state: state[np.newaxis, :]})
        return action[0]

def main(_):
    pass

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if type(tf.contrib) != type(tf): tf.contrib._warning = None
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument('--a_high', action='store', dest='a_high', help='Action space hgih')
    parser.add_argument('--a_low', action='store', dest='a_low', help='Action space low')
    parser.add_argument('--obs_dim', action='store', dest='obs_dim', help='Observation dimension')
    parser.add_argument('--opponent', action='store', dest='opponent', help='Directory of opponent')
    args = parser.parse_args()

    opponent = str(args.opponent)
    action_low, action_high = str(args.a_low).split('_'), str(args.a_high).split('_')
    action_space = spaces.Box(low=np.array(action_low), high=np.array(action_high), dtype=np.float32)
    obs_dim = int(args.obs_dim)
    high = np.inf*np.ones(obs_dim)
    low = -high
    observation_space = spaces.Box(low, high, dtype=np.float32)

    oppo = Infer(action_space.high, action_space.low, action_space.shape[0], observation_space.shape)
    ckpt = tf.train.latest_checkpoint(opponent)
    restorer = tf.train.import_meta_graph(ckpt + ".meta", clear_devices=True)
    sess = tf.Session()
    restorer.restore(sess, ckpt)
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    print("OK")
    while True:
        try:
            msg = str(input())
        except(EOFError):
            continue
        if msg == "terminate":
            break
        msg = msg.split(' ')
        msg = [float(m) for m in msg]
        action = oppo.evaluate_state(np.array(msg), sess)
        print(action[0], action[1])
