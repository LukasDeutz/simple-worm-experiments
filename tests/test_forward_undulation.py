# Build-in imports
from os.path import isfile

# Third party imports
import matplotlib.pyplot as plt

# Local imports
from simple_worm.plot3d import generate_interactive_scatter_clip
from simple_worm.plot3d_cosserat import plot_controls_CS_vs_FS, plot_single_strain_vs_control

from simple_worm_experiments.util import default_parameter
from simple_worm_experiments.forward_undulation.undulation import ForwardUndulation
from simple_worm_experiments.forward_undulation.plot_undulation import plot_trajectory

def test_forward_undulation():

    parameter = default_parameter()
    parameter['T'] = 1.0
    
    parameter['external_force'] = ['rft']

    E = 1e4
    G = E / (2 * (1 + 0.5))
    eta = 1e-2 * E 
    nu = 1e-2 * G

    parameter['E'] = E
    parameter['G'] = G
    parameter['eta'] = eta
    parameter['nu'] = nu
    parameter['rc'] = 'spheroid'
        
    # Kinematic parameter
    A = 4.0
    lam = 1.5
    f = 2.0
    
    parameter['A'] = A
    parameter['lam'] = lam
    parameter['f'] = f    
    parameter['smooth_muscle_onset'] = False
    
    # Fluid
    #gamma = 0.01    
    #parameter['gamma'] = gamma
    #parameter['c_t'] = 1.0
    #parameter['c_n'] = 1.6
    
    FU = ForwardUndulation(parameter['N'], parameter['dt'])
    
    FS, CS, _ = FU.simulate_undulation(parameter)
    
    CS = CS.to_numpy()
    
    generate_interactive_scatter_clip(FS, 500, perspective = 'xy', n_arrows = 50) # n_arrows= 65                               
    
    # plot_controls_CS_vs_FS(CS.to_numpy(), FS, parameter['dt'])    
    
    k0 = CS.Omega[:, 0, :]
    k = FS.Omega[:, 0, :]        
        
    plot_single_strain_vs_control(k0, k, dt = parameter['dt'], titles = [r'$\kappa_{2,0}$', r'$\kappa_{2}$'], cbar_format='%.1f', cmap = plt.get_cmap('plasma'))        
    plot_trajectory(FS, parameter)
    plt.show()
    
    return
                                  
if __name__ == "__main__":
    
    test_forward_undulation()
    
    print('Finished')

