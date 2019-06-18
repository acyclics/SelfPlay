import trueskill
import os
import re

class Player:
    def __init__(self, name, env):
        self.global_skill = trueskill.TrueSkill()
        self.global_skill.make_as_global()
        self.player_dir = ".\\players\\" + name
        with open(".\\leaderboard", "r") as file:
            players = file.readlines()
        found = False
        for p in players:
            if str(p) == name:
                found = True
        if found:
            self.load()
        else:
            players.append(name)
            with open(".\\leaderboard", "w") as file:
                file.writelines(players)
            self.new(name, env)

    def new(self, name, env):
        os.mkdir(self.player_dir)
        player_name = name
        player_env = env
        self.player_rating = self.global_skill.create_rating()
        details = ["Name: " + str(name) + "\n", "Env: " + str(env) + "\n"]
        with open(self.player_dir + "\\profile", "w") as file:
            file.writelines(details)
        self.record_player()

    def tmp_to_storage(self, PATH):
        os.mkdir(PATH)
        model = os.listdir(".\\players\\tmp")
        for f in model:
            shutil.move(".\\players\\tmp" + "\\" + f, PATH)

    def record_player(self):
        directories = self.sorted_directories()
        if len(directories) > 0:
            if self.player_rating > int(directories[-1]):
                ratings = int(directories[-1])
                PATH = "0"
                while ratings < self.player_rating:
                    ratings += 10
                    PATH =  self.player_dir + "\\" + str(ratings)
                    os.mkdir(PATH)
                PATH = PATH + "\\" + str(self.player_rating)
                self.tmp_to_storage(PATH)
            else:
                for i in range(len(directories)):
                    if self.player_rating <= int(directories[i]):
                        PATH =  self.player_dir + "\\" + str(directories[i]) + "\\" + str(self.player_rating)
                        self.tmp_to_storage(PATH)
        else:
            ratings = -10
            while ratings < self.player_rating:
                ratings += 10
                PATH = self.player_dir + "\\" + str(ratings)
                os.mkdir(PATH)
            PATH = PATH + "\\" + str(self.player_rating)
            self.tmp_to_storage(PATH)

    def load(self, rating_range):
        directories = self.sorted_directories()
        found = False
        for i in range(len(directories)):
            if rating_range <= int(directories[i]):
                files = self.sorted_files()
                if rating_range >= files[-1]:
                    found = True
                    self.player_rating = files[-1]
                for f in files:
                    if rating_range <= f:
                        found = True
                        self.player_rating = f
        if not found:
            self.player_rating = files[-1]
                
    def sorted_directories(self):
        directories = [int(d) for d in os.listdir(self.player_dir) if os.path.isdir(join(self.player_dir, d))]
        #directories = [int(d) for d in directories]
        return directories.sort()

    def sorted_files(self):
        files = [float(f) for f in os.listdir(self.player_dir) if os.path.isfile(join(self.player_dir, f))]
        return sort(files)
