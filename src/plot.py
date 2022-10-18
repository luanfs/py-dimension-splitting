####################################################################################
#
# Module for plotting routines.
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np
import matplotlib.pyplot as plt
from errors import *

####################################################################################
# Plot the 2d graphs given in the list fields
####################################################################################
def plot_2dfield_graphs(scalar_fields, scalar_fieldsmin, scalar_fieldsmax, xplot, yplot, vector_fieldsu, vector_fieldsv, xv, yv, filename, title):
    n = len(scalar_fields)
    for k in range(0, n):
        plt.contourf(xplot, yplot, scalar_fields[k], cmap='jet', levels=np.linspace(scalar_fieldsmin[k],scalar_fieldsmax[k],101))
        plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04)
        rnorm = 1.0
        #rnorm = np.sqrt(vector_fieldsu[k]**2 + vector_fieldsv[k]**2)
        plt.quiver(xv, yv, vector_fieldsu[k]/rnorm, vector_fieldsv[k]/rnorm)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.savefig(filename)
        plt.close()
