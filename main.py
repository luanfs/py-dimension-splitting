####################################################################################
#
# Dimension splitting methods
# Luan da Fonseca Santos - March 2022
#
###################################################################################
# Source code directory
srcdir = "src/"

import sys
import os.path
sys.path.append(srcdir)

# Imports
import configuration as conf
from miscellaneous           import createDirs

from advection_2d            import adv_2d
from parameters_2d           import simulation_par_2d
from error_adv_2d            import error_analysis_adv2d

# Create directories
createDirs()

# 2D advection test cases - parameters from par/configuration.par
# Get parameters
N, M, dt, Tf, tc, ic, mono = conf.get_test_parameters_2d('configuration.par')
simulation = simulation_par_2d(N, M, dt, Tf, ic, tc, mono)

# Select test case
if tc == 1:
    # Advection routine
    adv_2d(simulation, True)
elif tc == 2:
    # Advection error analysis
    error_analysis_adv2d(simulation)
else:
    print('Invalid testcase!\n')
    exit()
