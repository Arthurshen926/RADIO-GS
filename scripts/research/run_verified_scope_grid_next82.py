import subprocess,sys
py='/root/miniconda3/envs/cybersim_agent/bin/python'
for initialization in ['semantic','source']:
 subprocess.run([py,'scripts/research/run_verified_scope_next82.py','--scene',sys.argv[1],'--gpu',sys.argv[2],'--initialization',initialization],check=True)
