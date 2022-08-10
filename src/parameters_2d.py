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
def grid_1d(x0, xf, N):
    x  = np.linspace(x0, xf, N+1) # Cell edges
    xc = (x[0:N] + x[1:N+1])/2    # Cell centers
    dx = (xf-x0)/N                # Grid length
    return x, xc, dx

####################################################################################
# Create the 2d grid
####################################################################################
def grid_2d(x0, xf, N, y0, yf, M):
    x, xc, dx = grid_1d(x0, xf, N)
    y, yc, dy = grid_1d(y0, yf, M)
    return x, xc, dx, y, yc, dy

####################################################################################
#  Simulation class
####################################################################################
class simulation_par_2d:
    def __init__(self, N, M, dt, Tf, ic, tc, mono):
        # Number of cells in x direction
        self.N  = N

        # Number of cells in y direction
        self.M  = M

        # Initial condition
        self.ic = ic

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
            x0, xf = -1.0, 1.0
            y0, yf = -1.0, 1.0
            name = 'Sine wave'

        elif ic == 2:
            x0, xf = -1.0, 1.0
            y0, yf = -1.0, 1.0
            name = 'Gaussian wave'

        elif ic == 3:
            x0, xf = 0, 40
            y0, yf = 0, 40
            name = 'Triangular wave'

        elif ic == 4:
            x0, xf = 0, 40
            y0, yf = 0, 40
            name = 'Rectangular wave'

        elif ic == 5:
            x0, xf = -np.pi, np.pi
            y0, yf = -np.pi*0.5, np.pi*0.5
            name = 'Two gaussian hills'
        else:
            print("Error - invalid test case")
            exit()

        # Monotonization:
        if mono == 0:
            monot = 'none'
        elif mono == 1:
            monot = 'WC84' # Woodward and Collela 84 paper
        else:
           print("Error - invalid monotization method")
           exit()

        # Interval endpoints
        self.x0 = x0
        self.xf = xf
        self.y0 = y0
        self.yf = yf

        # Grid
        self.x, self.xc, self.dx, self.y, self.yc, self.dy = grid_2d(x0, xf, N, y0, yf, M)

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

####################################################################################
# Initial condition
####################################################################################
def q0_adv_2d(x, y, simulation):
    y = qexact_adv_2d(x, y, 0, simulation)
    return y

####################################################################################
# Exact solution to the advection problem
####################################################################################
def qexact_adv_2d(x, y, t, simulation):
    x0 = simulation.x0
    xf = simulation.xf
    y0 = simulation.y0
    yf = simulation.yf
    ic = simulation.ic

    if simulation.ic >= 1 and simulation.ic <= 4 : # constant speed
        u, v = velocity_adv_2d(x, y, t, simulation)
        X = x-u*t
        mask = (X != xf)
        X[mask] = (X[mask]-x0)%(xf-x0) + x0 # maps back to [x0,xf]

        Y = y-v*t
        mask = (Y != yf)
        Y[mask] = (Y[mask]-y0)%(yf-y0) + y0 # maps back to [y0,yf]

        if simulation.ic == 1:
            z = np.sin(2.0*np.pi*X)*np.sin(2.0*np.pi*Y) + 1.0

        elif simulation.ic == 2:
            z = np.exp(-5.0*(np.sin(np.pi*X))**2)
            z = z*np.exp(-5.0*(np.sin(np.pi*Y))**2)

        elif simulation.ic == 3:
            mask1 = np.logical_and(X>=15.0,X<=20.0)
            mask2 = np.logical_and(X>=20.0,X<=25.0)

            z = x*0
            z[mask1==True] = ( X[mask1==True] - 15.0)/5.0
            z[mask2==True] = (-X[mask2==True] + 25.0)/5.0

        elif simulation.ic == 4:
            maskx = np.logical_and(X>=15.0,X<=25.0)
            masky = np.logical_and(Y>=15.0,Y<=25.0)
            z = x*0
            z[np.logical_and(maskx, masky)] = 1.0

    elif simulation.ic == 5:
        A = 0.2
        Lx = 2*np.pi
        x0 = -1*Lx/12.0
        y0 = 0.0
        x1 =  1*Lx/12.0
        y1 = 0.0
        z0 = 0.95*np.exp(-((x-x0)**2 + (y-y0)**2)/A)
        z1 = 0.95*np.exp(-((x-x1)**2 + (y-y1)**2)/A)
        z = z0 + z1
    return z

####################################################################################
# Velocity field
####################################################################################
def velocity_adv_2d(x, y, t, simulation):
    if simulation.ic == 1:
        u = 0.2
        v = -0.1
    elif simulation.ic == 2:
        u = -0.1
        v = 0.1
    elif simulation.ic == 3:
        u = 0.5
        v = 0.5
    elif simulation.ic == 4:
        u = 0.5
        v = 0.5
    elif simulation.ic == 5:
        phi_hat = 10
        T = 5

        pi = np.pi
        twopi = pi*2.0
        Lx = twopi
        Ly = pi

        arg1 = twopi*(x/Lx - t/T)
        arg2 = pi*y/Ly
        arg3 = pi*t/T

        c1 = (phi_hat/T)*(Lx/(2*np.pi))**2

        #print(c1*4.0*pi/Lx)

        #print(c1*2.0*pi/Ly)

        #print(Lx/T)

        v = c1 * (2.0*pi/Lx) * (2.0*np.sin(arg1)*np.cos(arg1)) * (np.cos(arg2))**2 * np.cos(arg3)
        #print('v')
        #print(np.min(v), np.max(v))
        u = c1 * (pi/Ly) * (np.sin(arg1))**2 * (2.0*np.cos(arg2)*np.sin(arg2)) * (np.cos(arg3))
        #print(np.min(u), np.max(u))
        u = u - Lx/T
        u = -u
        v = -v
       # exit()
        #u = 0.5
        #v = 0.5
    return u, v
