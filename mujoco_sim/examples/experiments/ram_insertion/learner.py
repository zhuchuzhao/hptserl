import os 
import sys 
import subprocess 
 
# Set environment variables 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']  = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']  = '0.3'
 
# Build command with original arguments and pass through any additional args 
cmd = [
    'python', 
    '../../train_rlpd_copy.py', 
    '--exp_name=ram_insertion', 
    '--checkpoint_path=first_run',
    '--learner'
] + sys.argv[1:] 
 
# Execute the command 
subprocess.run(cmd) 