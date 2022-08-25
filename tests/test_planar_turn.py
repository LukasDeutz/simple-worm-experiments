# Build-in imports
from os.path import isfile
import pickle
from multiprocessing import Pool

# Third party imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from simple_worm.plot3d import generate_interactive_scatter_clip
from simple_worm.plot3d_cosserat import plot_controls_CS_vs_FS, plot_single_strain_vs_control

from simple_worm_experiments.util import default_parameter, get_solver
from simple_worm_experiments.planar_turn.planar_turn import PlanarTurnExperiment, wrap_simulate_planar_turn
from simple_worm_experiments.forward_undulation.plot_undulation import plot_trajectory

def get_base_parameter():
    
    # Simulation parameter
    N = 129
    dt = 0.01
    
    # Model parameter
    external_force = ['rft']
    rft = 'Whang'
    use_inertia = False
    
    # Solver parameter
    pi_alpha0_max = 0.9
    pi_maxiter = 1000
        
    # Geometric parameter    
    L0 = 1130 * 1e-6
    r_max = 32 * 1e-6
    rc = 'spheroid'
    eps_phi = 1e-3
    
    # Material parameter
    E = 1e5
    G = E / (2 * (1 + 0.5))
    eta = 1e-2 * E 
    nu = 1e-2 * G
    
    # Fluid 
    mu = 1e-3
    
    # Kinematic parameter
    A0 = 4.0
    lam0 = 1.5
    f0 = 2.0    
    
    A1 = 6.0
    lam1 = 1.5
    f1 = 2.0    
       
    T0 = 1.0 / f0
    T1 = 1.0 / f1
       
    T = round(4.0*T0, 1)
    t0 = 2*T0
    Delta_t = 0.5 * T1
        
    smo = True
    a_smo = 150
    du_smo = 0.05
        
    parameter = {}
             
    parameter['pi_alpha0'] = pi_alpha0_max            
    parameter['pi_maxiter'] = pi_maxiter             
                
    parameter['external_force'] = external_force
    parameter['use_inertia'] = use_inertia
    parameter['rft'] = rft    
    
    parameter['N']  = N
    parameter['dt'] = dt
    parameter['T'] = T
        
    parameter['L0'] = L0
    parameter['r_max'] = r_max
    parameter['rc'] = rc
    parameter['eps_phi'] = eps_phi
    
    parameter['E'] = E
    parameter['G'] = G
    parameter['eta'] = eta
    parameter['nu'] = nu
    parameter['mu'] = mu

    parameter['A0'] = A0
    parameter['lam0'] = lam0
    parameter['f0'] = f0
        
    parameter['A1'] = A1
    parameter['lam1'] = lam1
    parameter['f1'] = f1    
    
    parameter['T'] = T    
    parameter['t0'] = t0
    parameter['Delta_t'] = Delta_t   
        
        
    parameter['smo'] = smo
    parameter['a_smo'] = a_smo
    parameter['du_smo'] = du_smo
    
    return parameter
           
def test_planar_turn():
        
    parameter = get_base_parameter()
    
    #parameter['T']  = 2.0
    parameter['N']  = 257
    parameter['dt'] = 0.01
    
    parameter['dt'] = 0.01
    parameter['fdo'] = {1: 2, 2: 2}
    parameter['pi_alpha0_max'] = 0.7            
    parameter['pi_alpha0_min'] = 0.1
    parameter['pi_rel_err_growth_tol'] = 4.0
    parameter['A1'] = 6.0    
                
    PTE = PlanarTurnExperiment(parameter['N'], parameter['dt'], solver = get_solver(parameter), quiet = True)        
    FS, CS, _ = PTE.simulate_planar_turn(parameter, try_block=False)

    print(f'Picard iteration converged at every time step: {np.all(FS.pic)}')
    
    # fp = 'F0.pkl'        
    # F0 = pickle.load(open(fp, 'rb'))
    #
    # data_path = '/home/lukas/git/forward-worm/data/planar_turn/'
    # #wrap_simulate_planar_turn(parameter, data_path, 'test', True, save = 'all', _try = True, F0 = F0)
    
    # with open(fp, 'wb') as f:
    #     F0 = FS[-1]
    #     pickle.dump(F0, f)
                            
    #print(f'Picard iteration converged at every time step: {np.all(FS.pic)}')
            
    # CS.to_numpy()
    #
    # CS = CS.to_numpy()
    #
    # generate_interactive_scatter_clip(FS, 500, perspective = 'xy', n_arrows = 50) # n_arrows= 65                               
    
    # plot_controls_CS_vs_FS(CS.to_numpy(), FS, parameter['dt'])        
    # k0 = CS.Omega[:, 0, :]
    # k = FS.Omega[:, 0, :]        
    #
    # plot_single_strain_vs_control(k0, k, dt = parameter['dt'], titles = [r'$\kappa_{2,0}$', r'$\kappa_{2}$'], cbar_format='%.1f', cmap = plt.get_cmap('plasma'))        
    # plot_trajectory(FS, parameter)
    # plt.show()    
               
                                  
if __name__ == "__main__":
    
    test_planar_turn()
    
    print('Finished')

