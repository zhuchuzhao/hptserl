import os 
import sys 
import subprocess 
 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']  = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']  = '0.1'
 
cmd = [
    'python', '../../train_rlpd_copy.py', 
    '--exp_name=usb_pickup_insertion',
    '--checkpoint_path=../../experiments/usb_pickup_insertion/debug',
    '--actor'
] + sys.argv[1:] 
 
subprocess.run(cmd) 