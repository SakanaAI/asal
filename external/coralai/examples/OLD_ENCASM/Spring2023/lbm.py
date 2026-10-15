import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():
    """ Lattice Boltzmann Simulation """
    # Simulation parameters
    
    Nx                     = 400    # resolution x-dir
    Ny                     = 200    # resolution y-dir
    rho0                   = 100    # average density
    tau                    = 0.6    # collision timescale
    Nt                     = 1000   # number of timesteps
    plotRealTime = True # switch on for plotting as the simulation goes along

    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1

    # Initial Conditions
    F = np.ones((Ny,Nx,NL)) #* rho0 / NL
    np.random.seed(42)
    F += 0.1*np.random.randn(Ny,Nx,NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
    rho = np.sum(F,2)
    for i in idxs:
        F[:,:,i] *= rho0 / rho
    

    X, Y = np.meshgrid(range(Nx), range(Ny))
    center_x = Nx/4
    center_y = Ny/2
    obstacles = (X - center_x)**2 + (Y - center_y)**2 < (Ny/4)**2


    # obstacles = np.array([[False, False, False, False, False, False, False, False, False, False],
    #                             [False, False, False, False, False, False, False, False, False, False],
    #                             [False, False, False, False, False, False, False, False, False, False],
    #                             [False, False, False, False, False, False, False, False, False, False],
    #                             [False, True,  True,  True,  True,  True,  True,  True,  True,  False],
    #                             [False, False, False, False, False, True,  False, False, False, False],
    #                             [True,  True,  True,  True,  False, True,  False, False, False, False],
    #                             [False, False, False, False, False, True,  False, False, False, False],
    #                             [False, True,  True,  True,  True,  True,  False, False, False, False],
    #                             [False, False, False, False, False, False, False, False, False, False]])

    # obstacles = np.array(obstacles, dtype=np.bool)

    # Prep figure
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # Create 2x2 grid of subplots

    # Simulation Main Loop
    for it in tqdm(range(Nt)):

        # # Store the fluid variables at the cylinder's new location
        # old_cylinder = obstacles
        # F_old_location = F[old_cylinder,:].copy()

        # # Cylinder boundary
        # X, Y = np.meshgrid(range(Nx), range(Ny))
        # center_x = (Nx/4 + it/20) % Nx  # cylinder moves rightward
        # center_y = Ny/2
        # obstacles = (X - center_x)**2 + (Y - center_y)**2 < (Ny/4)**2
        # obstacles = ((X - Nx/4 - it/20) % Nx)**2 + (Y - Ny/2)**2 < (Nx/4 - it/20)**2  # cylinder moves rightward

        # # Add/subtract fluid at the cylinder's new/old positions
        # F[obstacles,:] += rho0 / NL  # add fluid at the cylinder's new position
        # F[old_cylinder,:] -= rho0 / NL  # subtract fluid at the cylinder's old position


        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
        
        # Set reflective boundaries
        bndryF = F[obstacles,:]
        bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]

        
        # Calculate fluid variables
        rho = np.sum(F,2)
        ux  = np.sum(F*cxs,2) / rho
        uy  = np.sum(F*cys,2) / rho
        
        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )

        F += -(1.0/tau) * (F - Feq)

        # Apply boundary 
        F[obstacles,:] = bndryF


        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % 10) == 0) or (it == Nt-1):
            plt.cla()

            for ax in axs.flatten():
                ax.clear()  # Clear previous plots
                ax.cla()

            ux[obstacles] = 0
            uy[obstacles] = 0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[obstacles] = np.nan
            vorticity = np.ma.array(vorticity, mask=obstacles)

            # Plot density
            im = axs[0, 0].imshow(rho, cmap='Blues')
            axs[0, 0].imshow(~obstacles, cmap='gray', alpha=0.3)  # Overlay cylinder
            # fig.colorbar(im, ax=axs[0, 0], orientation='horizontal')
            axs[0, 0].set_title('Density')

            # Plot vorticity
            im = axs[0, 1].imshow(vorticity, cmap='bwr')
            axs[0, 1].imshow(~obstacles, cmap='gray', alpha=0.3)  # Overlay cylinder
            # fig.colorbar(im, ax=axs[0, 1], orientation='horizontal')
            axs[0, 1].set_title('Vorticity')

            # Plot x velocity
            im = axs[1, 0].imshow(ux, cmap='bwr')
            axs[1, 0].imshow(~obstacles, cmap='gray', alpha=0.3)  # Overlay cylinder
            # fig.colorbar(im, ax=axs[1, 0], orientation='horizontal')
            axs[1, 0].set_title('x Velocity')

            # Plot y velocity
            im = axs[1, 1].imshow(uy, cmap='bwr')
            axs[1, 1].imshow(~obstacles, cmap='gray', alpha=0.3)  # Overlay cylinder
            # fig.colorbar(im, ax=axs[1, 1], orientation='horizontal')
            axs[1, 1].set_title('y Velocity')

            for ax in axs.flatten():
                ax.invert_yaxis()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_aspect('equal')
            
            plt.pause(0.001)


    # Save figure
    plt.savefig('latticeboltzmann.png',dpi=240)
    plt.show()

    return 0

if __name__== "__main__":
  main()
