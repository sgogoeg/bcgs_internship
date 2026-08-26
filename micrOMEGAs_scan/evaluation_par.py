'''
The script runs micromegas with the parameters specified in the input csv file.

Example:

python3 evaluation_par.py --input data/input.csv --output data/output.csv --micromegas-path /home/sgogoeg/heptools/micromegas_6.2.3/SingletDMDirectandIndirect --par-file-name data0.par

'''

import argparse
import csv
import subprocess
import os

# Para definir los parametros a manipular, debe corresponder con los nombres y el orden de los parámetros en el modelo entregado a micrOmegas

parser = argparse.ArgumentParser(description='Run Micromegas with specified parameters.')
parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--output', type=str, required=True, help='Path to the output CSV file')
parser.add_argument('--micromegas-path', type=str, required=True, help='Path to the Micromegas directory')
parser.add_argument('--par-file-name', type=str, required=True, help='Name of the .par file')

args = parser.parse_args()

params = ['MSDM', 'Lamb']

micrOMEGAs_path = args.micromegas_path
data_path = f"{args.micromegas_path}/{args.par_file_name}"
input_path = args.input
output_path = args.output

# Para limpiar el output cada vez que se corra el script
with open(output_path, 'w', newline='') as csvfile:
    pass 

# Writing the header
with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([params[0]+" (GeV)", " " + params[1], ' Omegah2', ' Annihilation SV (cm^3/s)',' SV bb (cm^3/s)', ' SV ww (cm^3/s)', ' SI-S (pb)']) 

# Para limpiar el data.par al principio
with open(data_path, 'w') as tempfile:
    pass

# Leyendo el input
with open(input_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    # Corriendo sobre las filas del input
    for row in reader:
        # Escribiendo los parametros en el data1.par
        with open(data_path, 'w') as tempfile:
            tempfile.write(f"{params[0]} {row[0]} \n{params[1]} {row[1]}\n")
        
        # Tomando el resultado
        result = subprocess.run(['./main', data_path], cwd = micrOMEGAs_path ,stdout=subprocess.PIPE, text=True)

        # Variable para almacenar la densidad de reliquia
        omega_value = None # Relic density
        SISigma = None # Direct detection spin-independent
        AnniSV = None # Indirect detection annihilation cross section
        AnniSVbb = 0 # Proportion of indirect detection annihilation cross section from b b~
        AnniSVww = 0 # Proportion of indirect detection annihilation cross section from W W~

        # Abriendo el output
        with open(output_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Buscando el valor en cada lines
            for line in result.stdout.split('\n'):
                if "Omega=" in line:
                    omega_value = line.split("Omega=")[1]
                if "annihilation cross section" in line:
                    split0 = line.split("annihilation cross section ")[1]
                    AnniSV = split0.split(" c")[0]
                if "~SDM,~SDM -> b b~" in line:
                    AnniSVbb = line.split("~SDM,~SDM -> b b~")[1]
                    AnniSVbb = AnniSVbb.strip()
                if "~SDM,~SDM -> W+ W-" in line:
                    AnniSVww = line.split("~SDM,~SDM -> W+ W-")[1]
                    AnniSVww = AnniSVww.strip()
                if "proton  SI" in line:
                    split1 = line.split("proton  SI ")[1]
                    SISigma = split1.split(" [")[0]
                    writer.writerow([row[0], row[1] ,omega_value, AnniSV, AnniSVbb, AnniSVww, SISigma]) 
                    break

os.remove(input_path)
                     