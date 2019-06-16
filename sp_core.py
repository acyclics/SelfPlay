'''
https://arxiv.org/pdf/1710.03748.pdf
'''
import numpy as np
import subprocess
from time import time, sleep
import datetime
import os
from sp_utils import read_settings, read_hyperparameters, write_settings

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
    def __init__(self):
        # Read data from save
        self.a = 1

    def process_interface(self):
        print("**************************************************\n",
              "********************SELF-PLAY*********************\n",
              "**************************************************\n")
        user_input = 1
        while (user_input):
            print("1 : Train\n",
                  "2 : Configure training\n",
                  "3 : Elo\n",
                  "0 : Exit\n",
                  "INPUT: ", end="")
            user_input = int(input())
            os.system('cls')
            self.process_handle(user_input)

    def process_handle(self, user_input):
        if user_input == 1:
            self.process_train()
        elif user_input == 2:
            self.process_configure()
        elif user_input == 0:
            return
        else:
            print("Invalid input")

    def process_configure(self):
        user_input = 1
        while (user_input):
            print("1 : Environment\n",
                  "2 : Number of workers\n",
                  "3 : Ignored gradients\n",
                  "4 : Parameter servers\n",
                  "5 : Name of output directory\n",
                  "6 : Number of episodes for each worker\n",
                  "7 : Gamma\n",
                  "8 : Lambda\n",
                  "9 : Timesteps till update\n",
                  "10 : Entropy beta\n",
                  "11 : Learning rate\n",
                  "12 : Size of minibatch\n",
                  "13 : Epochs\n",
                  "14 : Epsilon\n",
                  "15 : Value function coefficient\n",
                  "16 : L2 regularization\n",
                  "17 : Sigma floor\n",
                  "18 : Batch's buffer size\n",
                  "0 : Exit\n",
                  "INPUT: ", end="")
            user_input = int(input())
            if user_input == 0:
                os.system('cls')
                return
            else:
                self.process_configure_handle(user_input)

    def process_configure_handle(self, user_input):
        settings = read_settings()
        newValue = input("New value: ")
        os.system('cls')
        if user_input == 1:
            settings[0] = "EV=" + str(newValue) + "\n"
        elif user_input == 2:
            settings[1] = "WS=" + str(newValue) + "\n"
        elif user_input == 3:
            settings[2] = "DD=" + str(newValue) + "\n"
        elif user_input == 4:
            settings[3] = "PS=" + str(newValue) + "\n"
        elif user_input == 5:
            settings[4] = "OD=" + str(newValue) + "\n"
        elif user_input == 6:
            settings[5] = "ES=" + str(newValue) + "\n"
        elif user_input == 7:
            settings[6] = "GA=" + str(newValue) + "\n"
        elif user_input == 8:
            settings[7] = "LA=" + str(newValue) + "\n"
        elif user_input == 9:
            settings[8] = "US=" + str(newValue) + "\n"
        elif user_input == 10:
            settings[9] = "EB=" + str(newValue) + "\n"
        elif user_input == 11:
            settings[10] = "LR=" + str(newValue) + "\n"
        elif user_input == 12:
            settings[11] = "SB=" + str(newValue) + "\n"
        elif user_input == 13:
            settings[12] = "EP=" + str(newValue) + "\n"
        elif user_input == 14:
            settings[13] = "EN=" + str(newValue) + "\n"
        elif user_input == 15:
            settings[14] = "VF=" + str(newValue) + "\n"
        elif user_input == 16:
            settings[15] = "L2=" + str(newValue) + "\n"
        elif user_input == 17:
            settings[16] = "SF=" + str(newValue) + "\n"
        elif user_input == 18:
            settings[17] = "BB=" + str(newValue) + "\n"
        else:
            print("Invalid input")
        write_settings(settings)

    def process_train(self):
        hyperparameters = read_hyperparameters()
        N_WORKERS = int(hyperparameters[1])
        PS = int(hyperparameters[2])
        ts = time()
        TIMESTAMP = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
        processes = []
        with open(os.devnull, 'w') as tempf:
            for p in range(PS):
                cmd = "python algo_dppo.py --timestamp=" + str(TIMESTAMP) + " --job_name=\"ps\" --task_index=" + str(p)
                processes.append(subprocess.Popen(cmd, shell=True, stdout=tempf, stderr=tempf))
            for w in range(N_WORKERS):
                cmd = "python algo_dppo.py --timestamp=" + str(TIMESTAMP) + " --job_name=\"worker\" --task_index=" + str(w)
                if w == 0:
                    processes.append(subprocess.Popen(cmd, shell=True))
                else:
                    processes.append(subprocess.Popen(cmd, shell=True, stdout=tempf, stderr=tempf))

            processes[PS].wait()
            for p in processes:
                termination = subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=p.pid), stdout=tempf, stderr=tempf)
                termination.wait()

    def load_rating(self, player1, player2):
        return 1, 2

    def update_rating(self, winner, loser):
        return 1
