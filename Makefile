# -------------------------------------   Clear logs and cache  -------------------------------------
clear-logs:
	rm -rf logs/*

clear-wandb-cache:
	rm -rf wandb/*


# -------------------------------------   HPC-related   -------------------------------------
request-cpu-node:
	sintr -A MLMI-tw581-SL2-CPU -p icelake -N1 -n1 -t 1:00:00 --qos=INTR

request-gpu-node:
	sintr -A MLMI-tw581-SL2-GPU -p ampere --gres=gpu:1 -N1 -n1 -t 1:00:00 --qos=INTR

start-notebook-on-node:
	jupyter lab --no-browser --ip=* --port=8081

ssh-to-node-1:
	ssh -L 8081:gpu-q-20:8081 tw581@login-e-16.hpc.cam.ac.uk

# ssh-to-node-2:
# 	ssh -L 8081:localhost:8081 [INSERTNODE]
# [INSERTNODE] should be replaced with the node shown in squeue -u tw581, for instance gpu-q-63
