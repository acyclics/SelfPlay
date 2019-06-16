import subprocess
from time import time, sleep
import datetime
import os

def process_train():
    PS = 1
    N_WORKERS = 6
    N_DROPPED = 2
    N_AGG = N_WORKERS - N_DROPPED
    ts = time()
    TIMESTAMP = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = ".\\outputs"
    processes = []

    with open(os.devnull, 'w') as tempf:
        for p in range(PS):
            cmd = ("python algo_dppo.py --timestamp=" + str(TIMESTAMP) + " --job_name=\"ps\" --task_index=" + 
                    str(p) + " --workers=" + str(N_WORKERS) + " --agg=" + str(N_AGG) + " --ps=" + str(PS) +
                    " --output_dir=" + str(output_dir))
            processes.append(subprocess.Popen(cmd, shell=True, stdout=tempf, stderr=tempf))

        for w in range(N_WORKERS):
            cmd = ("python algo_dppo.py --timestamp=" + str(TIMESTAMP) + " --job_name=\"worker\" --task_index=" +
                    str(w) + " --workers=" + str(N_WORKERS) + " --agg=" + str(N_AGG) + " --ps=" + str(PS) +
                    " --output_dir=" + str(output_dir))
            if w == 0:
                processes.append(subprocess.Popen(cmd, shell=True))
            else:
                processes.append(subprocess.Popen(cmd, shell=True, stdout=tempf, stderr=tempf))

        processes[PS].wait()
        for p in processes:
            termination = subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=p.pid), stdout=tempf, stderr=tempf)
            termination.wait()
