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
from parameters_2d           import simulation_adv_par_2d
from advection_errors        import error_analysis_adv2d
from operator_accuracy       import error_analysis_div

# Create directories
createDirs()

# Get parameters
N, M, problem = conf.get_test_parameters_2d('configuration.par')

# Select test case
if problem == 1:
    # 2D advection test cases - parameters from par/configuration.par
    # Get parameters
    dt, Tf, tc, ic, vf, flux_method, dp, opsplit = conf.get_adv_parameters_2d('advection.par')
    simulation = simulation_adv_par_2d(N, M, dt, Tf, ic, vf, tc, flux_method, dp, opsplit)

    if tc == 1:
        # Advection routine
        adv_2d(simulation, True)
    elif tc == 2:
        # Advection error analysis
        error_analysis_adv2d(simulation)
    else:
        print('Invalid testcase!\n')
        exit()

elif problem == 2:
    print('Not implemented yet.')

elif problem == 3:
    # test the divergence error
    # Get parameters
    dt, Tf, tc, ic, vf, flux_method, dp, opsplit = conf.get_adv_parameters_2d('advection.par')
    simulation = simulation_adv_par_2d(N, M, dt, Tf, ic, vf, tc, flux_method, dp, opsplit)
    error_analysis_div(simulation)

else:
    print('Invalid testcase!\n')
    exit()
