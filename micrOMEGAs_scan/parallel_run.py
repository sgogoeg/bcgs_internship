'''
Runs exploration_par.py for every input file in the data folder and sends it to the background. 

Beware of many processes running at the same time, as it can cause problems with the system.

Example:

python3 parallel_run.py --micromegas-path /home/sgogoeg/heptools/micromegas_6.2.3/SingletDMDirectandIndirect

'''

import subprocess
import argparse
import glob
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Run exploration.py on existing input files in the current directory, log the process, and delete input files upon successful completion.')
parser.add_argument('--micromegas-path', type=str, required=True, help='Path to the Micromegas directory')
args = parser.parse_args()

# Path to Micromegas
micromegas_path = args.micromegas_path

# Find all input files in the inputs directory matching the pattern 'input_*.csv'
input_files = glob.glob(os.path.join('data', 'input_*.csv'))

dataparnum = 0

original_directory = os.getcwd()

for input_file in input_files:
    # Generate corresponding output filename by replacing 'inputs/input_' with 'outputs/output_' in the input filename
    output_file = os.path.join('data', 'output_' + os.path.basename(input_file).replace('input_', ''))
        
    # Construct the command to run exploration.py
    command = f'python3 evaluation_par.py --input {input_file} --output {output_file} --micromegas-path {micromegas_path} --par-file-name data{dataparnum}.par'
        
    nohup_command = f"nohup {command} >/dev/null 2>&1 &"

    dataparnum += 1

    # Run exploration.py with subprocess.run
    result = subprocess.run(nohup_command, shell=True, capture_output=False, text=False)
        
    # Determine success or failure
    status = 'SUCCESS' if result.returncode == 0 else 'FAILURE'
        

print("Process completed.")