####################################################################################
#
# Module for miscellaneous routines
#
# Luan da Fonseca Santos - April 2022
# (luan.santos@usp.br)
####################################################################################

import os
from parameters_2d import graphdir
####################################################################################
# Create a folder
# Reference: https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949
####################################################################################
def createFolder(dir):
   try:
      if not os.path.exists(dir):
         os.makedirs(dir)
   except OSError:
      print ('Error: Creating directory. '+ dir)

####################################################################################
# Create the needed directories
####################################################################################
def createDirs():
   print("--------------------------------------------------------")
   print("Dimension splitting python implementation by Luan Santos - 2022\n")
   # Check directory graphs does not exist
   if not os.path.exists(graphdir):
      print('Creating directory ',graphdir)
      createFolder(graphdir)

   print("--------------------------------------------------------")


