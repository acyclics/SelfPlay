import trueskill
import os
import re
import shutil
import tensorflow as tf

class Player:
    def __init__(self, name, env):
        self.global_skill = trueskill.TrueSkill()
        self.global_skill.make_as_global()
        self.player_dir = ".\\players\\" + name
        with open(".\\leaderboard", "r") as file:
            players = file.readlines()
        found = False
        for p in players:
            p = str(p).rstrip('\n')
            if p == name:
                found = True
                break
        if not found:
            players.append(name + '\n')
            with open(".\\leaderboard", "w") as file:
                file.writelines(players)
            self.new(name, env)

    def new(self, name, env):
        self.force_mkdir(self.player_dir)
        player_name = name
        player_env = env
        self.player_rating = self.global_skill.create_rating()
        details = ["Name: " + str(name) + "\n", "Env: " + str(env) + "\n"]
        with open(self.player_dir + "\\profile", "w") as file:
            file.writelines(details)
        self.record_player()

    def tmp_to_storage(self, PATH):
        self.force_mkdir(PATH)
        model = os.listdir(".\\players\\tmp")
        for f in model:
            shutil.move(".\\players\\tmp" + "\\" + f, PATH)

    def record_player(self):
        directories = self.sorted_directories()
        if len(directories) > 0:
            if self.player_rating.mu > int(directories[-1]):
                ratings = int(directories[-1])
                PATH = "0"
                while ratings < self.player_rating.mu:
                    ratings += 10
                    PATH =  self.player_dir + "\\" + str(ratings)
                    self.force_mkdir(PATH)
                PATH = PATH + "\\" + str(self.player_rating.mu)
                self.tmp_to_storage(PATH)
                self.save_trueskill(PATH)
                return
            else:
                for i in range(len(directories)):
                    if self.player_rating.mu <= int(directories[i]):
                        PATH =  self.player_dir + "\\" + str(directories[i]) + "\\" + str(self.player_rating.mu)
                        self.tmp_to_storage(PATH)
                        self.save_trueskill(PATH)
                        return
        else:
            ratings = -10
            while ratings < self.player_rating.mu:
                ratings += 10
                PATH = self.player_dir + "\\" + str(ratings)
                self.force_mkdir(PATH)
            PATH = PATH + "\\" + str(self.player_rating.mu)
            self.tmp_to_storage(PATH)
            self.save_trueskill(PATH)
            return

    def update(self, dir):
        skills = []
        skills.append(str(self.player_rating.mu) + '\n')
        skills.append(str(self.player_rating.sigma) + '\n')
        with open(os.path.join(dir, "skill"), "w") as file:
            file.writelines(skills)
        dst = os.path.join(os.path.dirname(dir), str(self.player_rating.mu))
        os.rename(dir, dst)
        if self.player_rating.mu > float(os.path.basename(os.path.dirname(dir))):
            directories = self.sorted_directories()
            for d in directories:
                if self.player_rating.mu <= d:
                    shutil.move(dst, os.path.join(self.player_dir, str(d)))
                    return
            ratings = int(directories[-1])
            while ratings < self.player_rating.mu:
                    ratings += 10
                    PATH =  os.path.join(self.player_dir, str(ratings))
                    self.force_mkdir(PATH)
            shutil.move(dst, os.path.join(self.player_dir, str(ratings)))

    def force_mkdir(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

    def load(self, rating_range):
        directories = self.sorted_directories()
        for i in range(len(directories)):
            if rating_range <= int(directories[i]):
                ratings = self.sorted_ratings(self.player_dir + "\\" + str(directories[i]))
                if len(ratings) == 0:
                    continue
                if rating_range >= ratings[-1]:
                    PLAY_DIR = self.player_dir + "\\" + str(directories[i]) + "\\" + str(ratings[-1])
                    self.player_rating = self.load_trueskill(PLAY_DIR)
                    if self.model_exist(PLAY_DIR):
                        return False, PLAY_DIR
                    else:
                        return True, PLAY_DIR
                for r in ratings:
                    if rating_range <= r:
                        PLAY_DIR = self.player_dir + "\\" + str(directories[i]) + "\\" + str(r)
                        self.player_rating = self.load_trueskill(PLAY_DIR)
                        if self.model_exist(PLAY_DIR):
                            return False, PLAY_DIR
                        else:
                            return True, PLAY_DIR
        for i in range(len(directories)):
            if rating_range <= int(directories[i]):
                ratings = self.sorted_ratings(self.player_dir + "\\" + str(directories[i]))
                if len(ratings) == 0:
                    OUT_DIR = self.player_dir + "\\" + str(directories[i]) + "\\" + str(self.player_rating.mu)
                    self.force_mkdir(OUT_DIR)
                    return True, OUT_DIR
                
    def sorted_directories(self):
        directories = [int(d) for d in os.listdir(self.player_dir) if os.path.isdir(os.path.join(self.player_dir, d))]
        directories.sort()
        return directories

    def sorted_ratings(self, path):
        ratings = [float(r) for r in os.listdir(path) if os.path.isdir(os.path.join(path, r))]
        ratings.sort()
        return ratings

    def highest_rating(self):
        directories = self.sorted_directories()
        directories.reverse()
        for d in directories:
            PATH = os.path.join(self.player_dir, str(d))
            ratings = self.sorted_ratings(PATH)
            ratings.reverse()
            for r in ratings:
                PLAY_DIR = os.path.join(PATH, str(r))
                if self.model_exist(PLAY_DIR):
                    return float(r), PLAY_DIR, self.load_trueskill(PLAY_DIR)
        directories.reverse()
        FIRST_DIR = os.path.join(self.player_dir, str(directories[-1]))
        ratings = self.sorted_ratings(FIRST_DIR)
        RATE_DIR = os.path.join(FIRST_DIR, str(ratings[-1]))
        return float(ratings[-1]), RATE_DIR, self.load_trueskill(RATE_DIR)

    def save_trueskill(self, directory):
        skill = [str(self.player_rating.mu) + '\n', str(self.player_rating.sigma) + '\n']
        with open(directory + "\\skill", "w") as file:
            file.writelines(skill)
    
    def load_trueskill(self, directory):
        with open(directory + "\\skill", "r") as file:
            skill = file.readlines()
        return self.global_skill.create_rating(float(skill[0]), float(skill[1]))

    def model_exist(self, directory):
        files = [str(f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for f in files:
            if "ckpt" in f:
                return True
        return False
