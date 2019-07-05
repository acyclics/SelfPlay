from game import Game
import numpy as np
import mujoco_py
from algo_dppo_infer import Infer

class test:
    def run(self):
        mode = "Runner"
        opponent = "C:\\Users\\impec\\Desktop\\Mods\\ai\\projects\\selfplay\\players\\chaser\\30\\25.740109390120463"
        first = True
        env = Game(5, 700, mode, opponent, first)
        obs = env.reset()
        action = [0, 10]
        for _ in range(500000):
            env.render()
            obs, rwd, done, info = env.step(action)
            if done:
                obs = env.reset()
        env.close()

debug = test()
debug.run()
