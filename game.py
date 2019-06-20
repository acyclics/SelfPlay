import mujoco_py as mjpy
import numpy as np
import cv2
from gym.utils import seeding
from gym import spaces
import subprocess
import os
from time import sleep
import tensorflow as tf

# CHANGABLE VARIABLES
DEFAULT_SIZE = 200

class Game:
    ''' Mujoco initialization '''
    def __init__(self, frame_skip, MAX_timestep, mode, opponent, first, rgb_rendering_tracking=True):
         # Mujoco specific
        xml_path = 'game.xml'
        self.model = mjpy.load_model_from_path(xml_path)
        self.sim = mjpy.MjSim(self.model)
        self.rgb_rendering_tracking = rgb_rendering_tracking
        # Model specific
        self.frame_skip = frame_skip
        self.data = self.sim.data
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.seed()
        self.viewer = None
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        # Data specific
        self.obs_dim = 10
        low = [-1, -1]
        high = [1, 1]
        self.action_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.timestep = 0
        self.totalTimeStep = 0
        self.MAX_timestep = MAX_timestep
        # multu-agent specific
        self.mode = mode
        self.first = first
        low = [-1, -1]
        high = [1, 1]
        low = [str(l) for l in low]
        high = [str(h) for h in high]
        low = '_'.join(low)
        high = '_'.join(high)
        if not first:
            cmd = "python algo_dppo_infer.py --a_high=" + high + " --a_low=" + low + " --obs_dim=" + str(self.obs_dim) + " --opponent=" + str(opponent)
            self.opponent_model = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            msg = ""
            while msg[0:4] != "b'OK":
                msg = str(self.opponent_model.stdout.readline())
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    ''' END '''




    ''' Model functions '''
    def read_model(self, s):
        s = [str(w) for w in s]
        s = ' '.join(s)
        self.opponent_model.stdin.write(bytes(str(s + "\r\n").encode(encoding='ascii')))
        self.opponent_model.stdin.flush()
        oppe_a = str(self.opponent_model.stdout.readline())
        oppe_a = oppe_a[2:-5].split(' ')
        oppe_a = [float(o) for o in oppe_a]
        return oppe_a
    def action_handle(self, s, a):
        if self.mode == "Chaser":
            a[0] = min(abs(a[0]), 5) * (a[0] / abs(a[0]))
            a[1] = min(abs(a[1]), 5) * (a[1] / abs(a[1]))
            self.sim.data.qvel[0:2] = a[0:2]
            if not self.first:
                oppe_a = self.read_model(s)
                self.sim.data.qvel[2:4] = oppe_a[0:2]
        else:
            self.sim.data.qvel[2:4] = a[0:2]
            if not self.first:
                oppe_a = self.read_model(s)
                oppe_a[0] = min(abs(oppe_a[0]), 5) * (oppe_a[0] / abs(oppe_a[0]))
                oppe_a[1] = min(abs(oppe_a[1]), 5) * (oppe_a[1] / abs(oppe_a[1]))
                self.sim.data.qvel[0:2] = oppe_a[0:2]
    ''' END '''




    ''' Model upper-level '''
    def step(self, a):
        done, reward, opponentR = self.reward_func()
        self.action_handle(self.pre_obs, a)
        self.do_simulation(self.frame_skip)
        self.timestep += 1
        self.totalTimeStep += 1
        ob = self._get_obs()
        self.pre_obs = ob
        if self.timestep >= self.MAX_timestep:
            done = True
        return ob, reward, done, opponentR
    def reward_func(self):
        vec = self.get_body_com("Runner") - self.get_body_com("Chaser")
        dist = np.linalg.norm(vec)
        eps = 2
        if self.mode == "Chaser":
            if dist <= eps:
                return True, 100000, -100000
            else:
                reward = -dist
        else:
            if dist <= eps:
                return True, -100000, 100000
            reward = dist
        return False, reward, -reward
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 30.0
    def reset_model(self):
        self.timestep = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()
    def _get_obs(self):
        return np.concatenate([
            self.get_body_com("Chaser"),
            self.sim.data.qvel[0:2],
            self.get_body_com("Runner"),
            self.sim.data.qvel[2:4]
        ])
    ''' END '''




    ''' Model lower-level '''
    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        self.pre_obs = ob
        return ob
    def set_state(self, qpos, qvel):
        old_state = self.sim.get_state()
        new_state = mjpy.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
    def do_simulation(self, n_frames):
        for _ in range(n_frames):
            self.sim.step()
    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'rgb_array':
            camera_id = None 
            camera_name = 'cam1'
            if self.rgb_rendering_tracking and camera_name in self.model.camera_names:
                camera_id = self.model.camera_name2id(camera_name)
            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()
    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}
        if not self.first:
            self.opponent_model.stdin.write(bytes(str("terminate" + "\r\n").encode(encoding='ascii')))
            self.opponent_model.stdin.close()
    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mjpy.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mjpy.MjRenderContextOffscreen(self.sim, -1)
                
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer
    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)
    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
    ''' END '''
