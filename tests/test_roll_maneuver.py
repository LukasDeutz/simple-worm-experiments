# Build-in imports
from os.path import isfile
import pickle
from multiprocessing import Pool

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Local imports
from simple_worm.plot3d import generate_interactive_scatter_clip
from simple_worm.plot3d_cosserat import plot_controls_CS_vs_FS, plot_single_strain_vs_control, plot_single_strain

from simple_worm_experiments.util import default_parameter, get_solver
from simple_worm_experiments.roll.roll import RollManeuverExperiment, wrap_simulate_roll_maneuver
from simple_worm_experiments.forward_undulation.plot_undulation import plot_trajectory

def get_base_parameter():
    
    # Simulation parameter
    N = 129
    dt = 0.01
    N_report = None
    dt_report = None
    
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
    A0 = 4.0 # Undulation amplitude
    lam0 = 1.5 # Undulation wavelength
    f0 = 2.0 # Undulation frequency
    
    #------------------------------------------------------------------------------ 
    # Relevant parameters to change for the roll maneuver
    
    # Roll maneuver parameter
    A1 = 8.0
    lam1 = 1.5
    f1 = 2.0    
       
    T0 = 1.0 / f0
    T1 = 1.0 / f1 
       
    T = round(4.0*T0, 1) # Simulation time
    
    t0 = 2*T0 # When the roll maneuver starts
    
    Delta_t = 0.5 * T1 # Roll maneuver duration

    #------------------------------------------------------------------------------ 
     
    # Sigmoid        
    smo = True
    a_smo = 150
    du_smo = 0.05
        
    parameter = {}
             
    parameter['N']  = N
    parameter['dt'] = dt
    parameter['T'] = T
    parameter['N_report'] = N_report
    parameter['dt_report'] = dt_report
    
    parameter['pi_alpha0'] = pi_alpha0_max            
    parameter['pi_maxiter'] = pi_maxiter             
                
    parameter['external_force'] = external_force
    parameter['use_inertia'] = use_inertia
    parameter['rft'] = rft    
        
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
           
def test_roll_maneuver():
        
    parameter = get_base_parameter()
    
    parameter['N']  = 257
    parameter['dt'] = 0.01
    parameter['pi'] = False
    
    pbar = tqdm.tqdm(desc = 'RME:')    
    RME = RollManeuverExperiment(parameter['N'], parameter['dt'], solver = get_solver(parameter), quiet = True)        
    FS, CS, _, e = RME.simulate_roll_maneuver(parameter, pbar)

    if e is not None:
        raise e

    print(f'Picard iteration converged at every time step: {np.all(FS.pic)}')
    
    plot_roll_maneuver(FS, CS, parameter)
    
    return
             
def plot_roll_maneuver(FS, CS, parameter):
    
    # Creates a video
    #generate_interactive_scatter_clip(FS, 500, n_arrows=25)

    # Plots all controls and strains
    plot_controls_CS_vs_FS(CS, FS, parameter['dt'])
    
    # Plot single control and strain
    k1_0 = CS.Omega[:, 0, :]
    k1 = FS.Omega[:, 0, :]    
    plot_single_strain_vs_control(k1_0, k1, dt = parameter['dt'])
    
    # Plot single strain or control
    plot_single_strain(k1, dt = parameter['dt'])
    
    plt.show()
    
    return
    
                                  
if __name__ == "__main__":
    
    test_roll_maneuver()
    
    print('Finished')

