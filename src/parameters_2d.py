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
    def __init__(self, N, M, dt, Tf, ic, vf, tc, recon, dp, opsplit):
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
        self.dt    = dt
        self.dto2  = dt*0.5
        self.twodt = dt*2.0

        # Total period definition
        self.Tf = Tf

        # Flux scheme
        self.recon = recon

        # Departure point scheme
        self.dp = dp

        # Operator splitting scheme
        self.opsplit = opsplit

        # Define the domain extremes, advection velocity, etc
        if ic == 1:
            name = 'Sine wave'

        elif ic == 2:
            name = 'Gaussian wave'

        elif ic == 3:
            name = 'Rectangular wave'

        elif ic == 4:
            name = 'Two gaussian hills'

        elif ic == 5:
            name = 'Constant field'

        else:
            print("Error - invalid initial condition", ic)
            exit()

        if vf >= 4:
            print("Error - invalid vector field", vf)
            exit()

        # Define the domain
        x0, xf = 0.0, 1.0
        y0, yf = 0.0, 1.0

        # Flux scheme
        if recon == 1:
            recon_name = 'PPM-0'
        elif recon == 2:
            recon_name = 'PPM-CW84' #Monotonization from Collela and Woodward 84 paper
        elif recon == 3:
            recon_name = 'PPM-PL07' #Hybrid PPM from Putman and Lin 07 paper
        elif recon == 4:
            recon_name = 'PPM-L04' #Monotonization from Lin 04 paper
        else:
           print("Error - invalid flux method")
           exit()

        # Departure point scheme
        if dp == 1:
            dp_name = 'RK1'
        elif dp == 2:
            dp_name = 'RK3'
        else:
           print("Error in simulation_adv_par_1d - invalid departure point scheme", dp)
           exit()

        # Operator scheme
        if opsplit == 1:
            opsplit_name = 'SP-AVLT' # Splitting from L96 paper (average lie-trotter splitting)
        elif opsplit == 2:
            opsplit_name = 'SP-L04' # Splitting from L04 paper
        elif opsplit == 3:
            opsplit_name = 'SP-PL07' #Splitting from Putman and Lin 07 paper
        else:
           print("Error - invalid operator splitting method")
           exit()

        self.opsplit_name = opsplit_name

        # Interval endpoints
        self.x0 = x0
        self.xf = xf
        self.y0 = y0
        self.yf = yf

        # Ghost cells variables
        if recon <= 3:
            self.ngl = 3
            self.ngr = 3
        elif recon == 4:
            self.ngl = 4
            self.ngr = 4

        self.ng  = self.ngl + self.ngr

        # Grid interior indexes
        self.i0   = self.ngl
        self.iend = self.ngl + N

        # Grid interior indexes
        self.j0   = self.ngl
        self.jend = self.ngl + M

        # Grid
        self.x, self.xc, self.dx, self.y, self.yc, self.dy = grid_2d(x0, xf, N, y0, yf, M, self.ngl, self.ngr, self.ng)

        # RK vars
        if dp == 3:
            self.K1u = np.zeros((N+self.ng+1,M+self.ng))
            self.K2u = np.zeros((N+self.ng+1,M+self.ng))
            self.K3u = np.zeros((N+self.ng+1,M+self.ng))
            self.K1u = np.zeros((N+self.ng,M+self.ng+1))
            self.K2u = np.zeros((N+self.ng,M+self.ng+1))
            self.K3u = np.zeros((N+self.ng,M+self.ng+1))

        # IC name
        self.icname = name

        # Flux method method
        self.recon_name = recon_name

        # Departure point method
        self.dp_name = dp_name

        # Simulation title
        if tc == 1:
            self.title = '2D Advection '
        elif tc == 2:
            self.title = '2D advection errors '
        else:
            print("Error - invalid test case")
            exit()


####################################################################################
#  Parabola class
####################################################################################
class ppm_parabola:
    def __init__(self, simulation, direction):
        # Number of cells
        N  = simulation.N
        M  = simulation.M
        ng = simulation.ng

        # reconstruction name
        self.recon_name = simulation.recon_name

        # parabola coefficients
        # Notation from Colella and  Woodward 1984
        # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
        self.q_L = np.zeros((N+ng, M+ng))
        self.q_R = np.zeros((N+ng, M+ng))
        self.dq  = np.zeros((N+ng, M+ng))
        self.q6  = np.zeros((N+ng, M+ng))

        if direction == 'x':
            # parabola fluxes
            self.f_L = np.zeros((N+ng+1, M+ng))  # flux from left
            self.f_R = np.zeros((N+ng+1, M+ng))  # flux from right
            self.f_upw = np.zeros((N+ng+1, M+ng)) # upwind flux

            # Extra variables for each scheme
            if simulation.recon_name == 'PPM-0' or simulation.recon_name == 'PPM-CW84' or simulation.recon_name == 'PPM-L04':
                self.Q_edges =  np.zeros((N+ng+1, M+ng))
        elif direction == 'y':
            # parabola fluxes
            self.f_L = np.zeros((N+ng, M+ng+1))   # flux from left
            self.f_R = np.zeros((N+ng, M+ng+1))   # flux from right
            self.f_upw = np.zeros((N+ng, M+ng+1)) # upwind flux

            # Extra variables for each scheme
            if simulation.recon_name == 'PPM-0' or simulation.recon_name == 'PPM-CW84' or simulation.recon_name == 'PPM-L04':
                self.Q_edges =  np.zeros((N+ng, M+ng+1))

        self.dF = np.zeros((N+ng, M+ng)) # div flux

        if simulation.recon_name == 'PPM-CW84':
            self.dQ  = np.zeros((N+ng, M+ng))
            self.dQ0 = np.zeros((N+ng, M+ng))
            self.dQ1 = np.zeros((N+ng, M+ng))
            self.dQ2 = np.zeros((N+ng, M+ng))

        if simulation.recon_name == 'PPM-L04':
            self.dQ      = np.zeros((N+ng, M+ng))
            self.dQ_min  = np.zeros((N+ng, M+ng))
            self.dQ_max  = np.zeros((N+ng, M+ng))
            self.dQ_mono = np.zeros((N+ng, M+ng))

####################################################################################
#  Velocity class
#  The quadrilateral points are labeled as below
#
#  po-------pv--------po
#  |                  |
#  |                  |
#  |                  |
#  pu       pc        pu
#  |                  |
#  |                  |
#  |                  |
#  po--------pv-------po
#
####################################################################################
class velocity_edges:
    def __init__(self, simulation, pos):
        N = simulation.N
        M = simulation.M
        ng = simulation.ng

        if pos == 'pu':
            self.u = np.zeros((N+1+ng, M+ng))
            self.v = np.zeros((N+1+ng, M+ng))
            self.u_averaged = np.zeros((N+1+ng, M+ng)) # used for departure point
            self.u_old      = np.zeros((N+1+ng, M+ng)) # used for departure point
        elif pos == 'pv':
            self.u = np.zeros((N+ng, M+1+ng))
            self.v = np.zeros((N+ng, M+1+ng))
            self.v_averaged = np.zeros((N+ng, M+1+ng)) # used for departure point
            self.v_old      = np.zeros((N+ng, M+1+ng)) # used for departure point

        else:
            print('ERROR in  velocity class: invalid position, ', pos)
