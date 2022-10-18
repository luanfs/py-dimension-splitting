####################################################################################
#
# Module for 1D test case set up (grid, initial condition, exact solution and etc)
#
# Luan da Fonseca Santos - April 2022
# (luan.santos@usp.br)
####################################################################################
import numpy as np

# Directory parameters
graphdir = "graphs/"            # Graphs directory
pardir   = "par/"               # Parameter files directory

####################################################################################
# Create the 1d grid
####################################################################################
def grid_1d(x0, xf, N, ngl, ngr, ng):
    dx  = (xf-x0)/N                # Grid length
    x   = np.linspace(x0-ngl*dx, xf+ngr*dx, N+1+ng) # Cell edges
    xc  = (x[0:N+ng] + x[1:N+1+ng])/2    # Cell centers
    return x, xc, dx

####################################################################################
# Create the 2d grid
####################################################################################
def grid_2d(x0, xf, N, y0, yf, M, ngl, ngr, ng):
    x, xc, dx = grid_1d(x0, xf, N, ngl, ngr, ng)
    y, yc, dy = grid_1d(y0, yf, M, ngl, ngr, ng)
    return x, xc, dx, y, yc, dy

####################################################################################
#  Simulation class
####################################################################################
class simulation_adv_par_2d:
    def __init__(self, N, M, dt, Tf, ic, vf, tc, mono):
        # Number of cells in x direction
        self.N  = N

        # Number of cells in y direction
        self.M  = M

        # Initial condition
        self.ic = ic

        # Velocity field
        self.vf = vf

        # Test case
        self.tc = tc

        # Time step
        self.dt = dt

        # Total period definition
        self.Tf = Tf

        # Monotonization
        self.mono = mono

        # Define the domain extremes, advection velocity, etc
        if ic == 1:
            name = 'Sine wave'

        elif ic == 2:
            name = 'Gaussian wave'

        elif ic == 3:
            name = 'Rectangular wave'

        elif ic == 4:
            name = 'Two gaussian hills'
        else:
            print("Error - invalid initial condition")
            exit()

        # Define the domain
        x0, xf = -np.pi, np.pi
        y0, yf = -np.pi*0.5, np.pi*0.5

        # Monotonization:
        if mono == 0:
            monot = 'none'
        elif mono == 1:
            monot = 'CW84' # Collela and Woodward 84 paper
        else:
           print("Error - invalid monotization method")
           exit()

        # Interval endpoints
        self.x0 = x0
        self.xf = xf
        self.y0 = y0
        self.yf = yf

        # Ghost cells variables
        self.ngl = 3
        self.ngr = 3
        self.ng  = self.ngl + self.ngr

        # Grid interior indexes
        self.i0   = self.ngl
        self.iend = self.ngl + N
        # Grid interior indexes
        self.j0   = self.ngl
        self.jend = self.ngl + M

        # Grid
        self.x, self.xc, self.dx, self.y, self.yc, self.dy = grid_2d(x0, xf, N, y0, yf, M, self.ngl, self.ngr, self.ng)

        # IC name
        self.icname = name

        # Monotonization method
        self.monot = monot

        # Finite volume method
        self.fvmethod = 'PPM'

        # Simulation title
        if tc == 1:
            self.title = '2D Advection '
        elif tc == 2:
            self.title = '2D advection errors '
        else:
            print("Error - invalid test case")
            exit()

