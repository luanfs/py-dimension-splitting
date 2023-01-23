####################################################################################
#
# This module contains all the routines that get the needed
# parameters from the /par directory.
#
# Luan da Fonseca Santos - April 2022
# (luan.santos@usp.br)
####################################################################################

import os.path
from parameters_2d import pardir

def get_test_parameters_2d(filename):
    # The standard file filename.par must exist in par/ directory
    file_path = pardir+filename

    if os.path.exists(file_path): # The file exists
        # Open the grid file
        confpar = open(file_path, "r")

        # Read the grid file lines
        confpar.readline()
        confpar.readline()
        N = confpar.readline()
        confpar.readline()
        M = confpar.readline()
        confpar.readline()
        eq = confpar.readline()
        confpar.readline()

        # Close the file
        confpar.close()

        # Convert from str to int
        N  = int(N)
        M  = int(M)
        eq = int(eq)

        if eq == 1:
            eqname = 'Advection'
        elif eq == 2:
            eqname = 'Shallow-water'

        #Print the parameters on the screen
        print("\n--------------------------------------------------------")
        print("Parameters from file", file_path,"\n")
        print("Number of cells in x direction: ", N)
        print("Number of cells in y direction: ", M)
        print("Equation: ", eqname)
        print("--------------------------------------------------------\n")


    else:   # The file does not exist
        print("ERROR in get_grid_parameters: file "+ filename +" not found in /par.")
        exit()
    return N, M, eq

def get_adv_parameters_2d(filename):
    # The standard file filename.par must exist in par/ directory
    file_path = pardir+filename

    if os.path.exists(file_path): # The file exists
        # Open the grid file
        confpar = open(file_path, "r")

        # Read the grid file lines
        confpar.readline()
        confpar.readline()
        Tf = confpar.readline()
        confpar.readline()
        dt = confpar.readline()
        confpar.readline()
        ic = confpar.readline()
        confpar.readline()
        vf = confpar.readline()
        confpar.readline()
        tc = confpar.readline()
        confpar.readline()
        recon = confpar.readline()
        confpar.readline()
        opslit = confpar.readline()
        confpar.readline()


        # Close the file
        confpar.close()

        # Convert from str to int
        Tf = float(Tf)
        dt = float(dt)
        ic = int(ic)
        vf = int(vf)
        tc = int(tc)
        recon = int(recon)
        opslit = int(opslit)

        #Print the parameters on the screen
        print("\n--------------------------------------------------------")
        print("Parameters from file", file_path,"\n")
        print("Time step: ", dt)
        print("Initial condition: ", ic)
        print("Velocity field: ", vf)
        print("Reconstruction method: ", recon)
        print("Operator splitting: ", opslit)
        print("--------------------------------------------------------\n")

    else:   # The file does not exist
        print("ERROR in get_grid_parameters: file "+ filename +" not found in /par.")
        exit()
    return  dt, Tf, tc, ic, vf, recon, opslit
