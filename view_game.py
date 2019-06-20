from game import Game
import numpy as np
import mujoco_py
from algo_dppo_infer import Infer

class test:
    def run(self):
        mode = "Chaser"
        opponent = "C:\\Users\\impec\\Desktop\\Mods\\ai\\projects\\selfplay\\players\\runner\\30\\24.954246683290606"
        first = False
        env = Game(5, 10000, mode, opponent, first)
        obs = env.reset()
        action = [1, 1]
        for _ in range(500000):
            env.render()
            obs, rwd, done, info = env.step(action)
            if done:
                obs = env.reset()
        env.close()

debug = test()
debug.run()
