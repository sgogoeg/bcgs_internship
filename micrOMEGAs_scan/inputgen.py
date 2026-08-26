'''
The script generates a csv file with selected points within a specified parameter space in logarithmic scale.
'''

import numpy as np

# Number of points to generate

num_samples = 10

startcoup = -3
stopcoup =  0
basecoup = 10 

# Couplings

couplings = np.logspace(startcoup, stopcoup, num_samples, base=basecoup)

# Masses

startmass = 0
stopmass =  3
basemass = 10 

masses = np.logspace(startmass, stopmass, num_samples, base=basemass)

input = []

for i in masses:

    mass = i*np.ones(couplings.shape)
    inputtemp = np.column_stack((mass, couplings))

    input.append(inputtemp)

input = np.vstack(input)

# Saving the input to a CSV file
np.savetxt("data/input.csv", input, delimiter=",")