def read_settings():
    with open("./settings", "r") as file:
        settings = file.readlines()
    return settings

def read_hyperparameters():
    settings = read_settings()
    for i in range(len(settings)):
        settings[i] = settings[i][3:-1]
    return settings

def write_settings(settings):
    with open("./settings", "w") as file:
        file.writelines(settings)
        