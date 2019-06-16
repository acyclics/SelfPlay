import trueskill
import os

class Player:
    def __init__(self, name, env):
        self.global_skill = trueskill.TrueSkill()
        self.global_skill.make_as_global()
        with open("./leaderboard", "r") as file:
            players = file.readlines()
        found = False
        for p in players:
            if str(p) == name:
                found = True
        if found:
            self.load()
        else:
            players.append(name)
            with open("./leaderboard", "w") as file:
                file.writelines(players)
            self.new(name, env)

    def new(self, name, env):
        path = "./players/" + name
        os.mkdir(path)
        player_name = name
        player_env = env
        player_rating = self.global_skill.create_rating()
        details = ["Name: " + str(name) + "\n", "Env: " + str(env) + "\n"]
        with open(path + "/profile", "w") as file:
            file.writelines(details)

    def load(self):
        return 1

p = Player("gimbal", "gimbal")
