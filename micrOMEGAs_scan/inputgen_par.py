'''
The script generates a set of csv files with selected points within a specified parameter space in logarithmic scale.

Example:

python3 inputgen_par.py --output data/input.csv --pointsMass 50 --pointsCoupling 50 --minMass 1e0 --maxMass 1e3 --minCoupling 1e-3 --maxCoupling 1e0 --files 4

'''

import argparse
import csv
import numpy as np
import os

parser = argparse.ArgumentParser(description='Generates desired parameter space.')

parser.add_argument('--output', type=str, required=True, help='Path to the output CSV file')

parser.add_argument('--pointsMass', type=int, default=1000, help='Number of points in Mass')
parser.add_argument('--pointsCoupling', type=int, default=1000, help='Number of points in the desired coupling')

parser.add_argument('--minMass', type=float, default=1, help='Lower limit on Mass in GeV')
parser.add_argument('--maxMass', type=float, default=1000, help='Upper limit on Mass in GeV')

parser.add_argument('--minCoupling', type=float, default=1, help='Lower limit on Coupling')
parser.add_argument('--maxCoupling', type=float, default=1000, help='Upper limit on Coupling')

parser.add_argument('--files', type=int, required=True, help='Number of files to split output')

args = parser.parse_args()

# Parametros generales del espaciado

base = 10

# Mass

Mass = np.logspace(np.log10(args.minMass), np.log10(args.maxMass), args.pointsMass, base=base)

# Lambda

coupling = np.logspace(np.log10(args.minCoupling), np.log10(args.maxCoupling), args.pointsCoupling, base=base)

input = []

for mass in Mass:
    for c in coupling:
        input_temp = np.array([mass, c])
        input.append(input_temp)

# Convirtiendo en un solo array de numpy y guardando

input = np.vstack(input)

np.savetxt(f'{args.output}', input, delimiter=",")

# Ahora el splitting del output

def split_csv(input_file, n):
    # Open the input CSV file
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        
        # Read the header row
        #header = next(reader)
        
        # Calculate the number of rows per new file
        total_rows = sum(1 for _ in reader)
        rows_per_file = total_rows // n
        
        # Reset the file pointer to the beginning
        infile.seek(0)
        #next(reader)  # Skip the header again
        
        # Create and write to new CSV files
        for i in range(n):
            with open(f"{input_file[:-4]}_part_{i}.csv", 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                
                # Write the header row
                #writer.writerow(header)
                
                # Write rows for this file
                for _ in range(rows_per_file):
                    try:
                        writer.writerow(next(reader))
                    except StopIteration:
                        break
                
                # Write remaining rows for the last file
                if i == n - 1:
                    writer.writerows(reader)
    # Delete the original file
    try:
        os.remove(input_file)
    except OSError as e:
        print(f"Error deleting original file '{input_file}': {e}")

split_csv(f'{args.output}', args.files)