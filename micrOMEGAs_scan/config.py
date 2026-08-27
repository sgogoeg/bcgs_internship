'''
Hand-edited configuration for the relic-line scan.

Everything that depends on where micrOMEGAs lives, what the executable is
called, or how the model card names its parameters is collected here.  No
other file in this package hardcodes a path or a parameter name.
'''

import math
from pathlib import Path

# ---------------------------------------------------------------- paths

# Directory containing the compiled micrOMEGAs executable.  The executable is
# run with this as its working directory, exactly as in micrOMEGAs_scan.
MICROMEGAS_PATH = Path('/home/sgogoeg/heptools/micromegas_7.1.4/DarkPhotonComplexScalar')

# Name of the executable inside MICROMEGAS_PATH.
EXECUTABLE = 'scan_omega'

# Name of the parameter file written before each call.  It is created inside
# MICROMEGAS_PATH and passed to the executable as an absolute path.  Give each
# concurrent scan its own name if this is ever parallelised.
PAR_FILE_NAME = 'scan.par'

# ------------------------------------------------------- model parameters

# Names of the four fields written to the .par file, one per line, as
# "<name> <value>".  Change these if the model card is renamed.
PAR_NAMES = {
    'mchi': 'Mchi',   # dark matter mass, GeV
    'map':  'MAp',    # dark photon mass, GeV
    'eps':  'epsD',   # kinetic mixing
    'gD':   'gD',     # dark gauge coupling
}

# Fixed during the scan.  Only MAp and epsD are scanned.
ALPHA_D = 0.1           # dark fine structure constant
MCHI_OVER_MAP = 0.6     # Mchi = MCHI_OVER_MAP * MAp

# The .par file takes gD, which is derived from ALPHA_D below.  If you would
# rather fix gD directly, set it here and it wins over ALPHA_D.
GD = None

# ------------------------------------------------------------- target

# Observed relic abundance.  The scan solves Omega h^2 = OMEGA_TARGET.
OMEGA_TARGET = 0.12

# ------------------------------------------------------------ execution

# Seconds allowed per micrOMEGAs call before it is treated as a failed point.
TIMEOUT = 300.0

# Where results and the evaluation cache are written.
DATA_DIR = Path(__file__).resolve().parent / 'data'


def gD_value():
    '''gD to write to the .par file.

    g_D = sqrt(4 pi alpha_D) unless GD was set explicitly.
    '''
    if GD is not None:
        return float(GD)
    return math.sqrt(4.0 * math.pi * ALPHA_D)
