import numpy as np
import matplotlib.pyplot as plt
from time import time as ti
from matplotlib import cm
from tqdm import tqdm

class lbm:
    def __init__(self, nx, ny, Re, tau, v, w, rho0, u0, omega = None):
        self.Re = Re                            # Reynolds number
        #------------------------------------------------------------------------------
        # self.maxIter = 1000
        self.nx, self.ny = nx, ny               # Domain dimensions
        self.ly = ny-1
        self.uLB = 0.04                         # Inlet velocity NON PHYSICAL??
        self.cx, self.cy, self.r = nx//4, ny//2, ny/9   # cylinder coordinates and radius (as integers)
        self.nulb = self.uLB*self.r/self.Re     # Viscosity
        if omega is not None:
            self.omega = 1 / (3*self.nulb+0.5)  # Relaxation parameter
        self.omega = 1 / (3*self.nulb+0.5)  

        # lattice velocities
        self.v = np.array([ 
                    [1,1],
                    [1,0],
                    [1,-1],
                    [0,1],
                    [0,0],
                    [0,-1],
                    [-1,1],
                    [-1,0],
                    [-1,-1]
                    ])

        # weights
        self.t = np.array([ 
                    1/36,
                    1/9,
                    1/36,
                    1/9,
                    4/9,
                    1/9,
                    1/36,
                    1/9,
                    1/36
                    ])

        self.col_0 = np.array([0,1,2])
        self.col_1 = np.array([3,4,5])
        self.col_2 = np.array([6,7,8])

# instantiate the cylindrical obstacle
obstacle = np.fromfunction(obstacle_fun(cx,cy,r),(nx, ny))
if True:
  plt.imshow(obstacle)

# initial velocity profile
vel = np.fromfunction(inivel(uLB,ly),(2,nx,ny))

# initialize fin to equilibirum (rho = 1)
fin = equilibrium(1,vel,v,t,nx,ny)

    def macroscopic(fin, nx, ny, v):
        """Extract macroscopic parameters
    
        This function returns the macroscopic variables density (rho = $\rho$;
        rank-2 tensor $\rightarrow$ scalar at every discrete lattice point) and
        velocity (u; rank-3 tensor x and y velocity at every discrete lattice point). 
        """
        rho = np.sum(fin,axis=0)
        u = np.zeros((2,nx,ny))
        for i in range(9):
            u[0,:,:] += v[i,0]*fin[i,:,:]
            u[1,:,:] += v[i,1]*fin[i,:,:]
        u /= rho
        return rho, u

    def equilibrium(rho, u, v, t, nx, ny):
        """Compute equilibrium distribution

        """

        usqr = (3/2)*(u[0]**2+u[1]**2)
        feq = np.zeros((9,nx,ny))
        for i in range(9):
            cu = 3*(v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
            feq[i,:,:] = rho*t[i]*(1+cu+0.5*cu**2-usqr)
        return feq

    def obstacle_fun(cx, cy, r):
        """
        ## Flow obstacle

        The numpy function *fromfunction* uses this function to broadcast the coordinates
        of the 'obstacle.' This currying is just to make a function that doesn't need any
        global variables. You can think about how you would use something like porespy or
        a CT scan to generate a porous medium that you could substitute for this function.
        """
        def inner(x, y):
            return (x-cx)**2+(y-cy)**2<r**2
        return inner


    def inivel( uLB, ly):
        """
        ## Inlet velocity

        This function instantiates the inlet boundary velocity (with a small perturbation).
        """
        def inner(d,x,y):
            return (1-d) * uLB * (1+1e-4*np.sin(y/ly*2*np.pi))
        return inner


