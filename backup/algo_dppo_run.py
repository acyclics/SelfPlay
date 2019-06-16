import subprocess
from time import time
import datetime

if __name__ == "__main__":
    PS = 1
    N_WORKERS = 6
    N_DROPPED = 2
    N_AGG = N_WORKERS - N_DROPPED
    ts = time()
    TIMESTAMP = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    for p in range(PS):
        cmd = "python algo_dppo.py --timestamp=" + str(TIMESTAMP) + " --job_name=\"ps\" --task_index=" + str(p) + " --workers=" + str(N_WORKERS) + " --agg=" + str(N_AGG) + " --ps=" + str(PS)
        proc = subprocess.Popen(cmd, shell=True)
        
    for w in range(N_WORKERS):
        cmd = "python algo_dppo.py --timestamp=" + str(TIMESTAMP) + " --job_name=\"worker\" --task_index=" + str(w) + " --workers=" + str(N_WORKERS) + " --agg=" + str(N_AGG) + " --ps=" + str(PS)
        proc = subprocess.Popen(cmd, shell=True)
