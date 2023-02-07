# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from pathlib import Path

# Local imports
from simple_worm.plot3d import generate_interactive_scatter_clip
from simple_worm.plot3d_cosserat import plot_CS_vs_FS, plot_S, plot_multiple_scalar_fields

from simple_worm_experiments.util import get_solver 
from simple_worm_experiments.roll.roll import RollManeuverExperiment
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm_experiments.worm_studio import WormStudio
from simple_worm_experiments.experiment import simulate_experiment, init_worm

#------------------------------------------------------------------------------ 
# Parameter

def base_parameter():
        
    parameter = {}
               
    # Simulation parameter
    parameter['N']  = 100
    parameter['dt'] = 0.01
    parameter['N_report'] = 100
    parameter['dt_report'] = 0.01      
    
    # Solver parameter
    parameter['pi'] = False
    parameter['pi_alpha0'] = 0.9            
    parameter['pi_maxiter'] = 1000             
    parameter['fdo'] = {1: 2, 2:2}

    # Model parameter
    parameter['use_inertia'] = False
    
    # Fluid parameter                
    parameter['external_force'] = ['rft']
    parameter['rft'] = 'Whang'
    parameter['mu'] = 1e-3
     
    # Geometric parameter            
    parameter['L0'] = 1130 * 1e-6
    parameter['r_max'] = 32 * 1e-6
    parameter['rc'] = 'spheroid'
    parameter['eps_phi'] = 1e-3
        
    # Material parameter
    parameter['E'] = 1e5
    parameter['G'] = parameter['E'] / (2 * (1 + 0.5)) / 10
    parameter['eta'] = 1e-2 * parameter['E'] 
    parameter['nu'] = 1e-2 * parameter['G']
            
    parameter['mts'] = True
    parameter['tau_on'] = 0.01
    parameter['tau_off'] = 0.01    
    parameter['Dt_on'] = 3*parameter['tau_on']
    parameter['Dt_off'] = 3*parameter['tau_off']
        
    return parameter

def continuous_roll_parameter():

    parameter = base_parameter()
    
    parameter['A_dv'] = 6.0
    parameter['f_dv'] = 5.0    

    parameter['A_lr'] = 6.0
    parameter['f_lr'] = 5.0    
    parameter['phi'] = np.pi/2

    parameter['Ds_h'] = None
    parameter['s0_h'] = None

    parameter['Ds_t'] = 1.0/32
    parameter['s0_t'] = 0.25
    
    return parameter

def single_roll_parameter():
    
    parameter = base_parameter()

    parameter['T'] = 1.0
        
    # Kinematic parameter        
    parameter['A_static'] = 3.0
    parameter['lam_static'] = 1.5 
    parameter['T_init'] = 0.5

    parameter['A_dv'] = 5.0
    parameter['f_dv'] = 8.0    
    parameter['A_lr'] = 5.0
    parameter['f_lr'] = 8.0    
    parameter['phi'] = np.pi/2
    
    # Roll duration
    parameter['Dt_roll'] = 0.2 / parameter['f_dv']

    # Muscle parmaeter    
    parameter['Ds_h'] = 1.0/16
    parameter['s0_h'] = 0.3
    
    return parameter

#------------------------------------------------------------------------------ 
# "Test" functions
      
def test_continuous_roll_maneuver(
        plot_control = False,
        make_video = True, 
        active_clip = True, 
        plot = True):
        
    parameter = continuous_roll_parameter()
    parameter['T'] = 5.0
    parameter['dt'] = 0.01
    parameter['N'] = 100    
    parameter['A_dv'] = 6.0
    
    pbar = tqdm.tqdm(desc = 'RME:')    
    RME = RollManeuverExperiment()        
    
    worm = init_worm(parameter)

    CS = RME.continuous_roll(worm, parameter)
    
    if plot_control:
        plot_S(CS.to_numpy())
        plt.show()
        
    FS, CS, _, e = simulate_experiment(worm, parameter, CS, pbar = pbar)
            
    if e is not None:
        raise e

    print(f'Picard iteration converged at every time step: {np.all(FS.pic)}')

    alpha = FS.theta[:, 0, :]  
    t = FS.times         
    f_avg, f_std = EPP.comp_roll_frequency_from_euler_angle(alpha, t, Dt = 0.5)
                          
    print(f'f_avg: {f_avg:.2f}')
    print(f'f_std: {f_std:.3f}')
                        
    if make_video:        
        WS = WormStudio(FS)        
        output_path = Path(
            f'../videos/continuous_A_{parameter["A_dv"]}_f_{parameter["f_dv"]}_s0_h={parameter["s0_h"]}_s0_t={parameter["s0_t"]}_roll')
        WS.generate_clip(output_path, 
            add_trajectory = False, 
            add_frame_vectors = True,
            draw_e3 = False,
            n_arrows = 0.2)        
    if active_clip:
        generate_interactive_scatter_clip(FS, 500, n_arrows=25)    
    if plot:
        plot_roll_maneuver(FS, CS, parameter)
            
    return
    
# TODO 
def test_single_roll_maneuver():

    parameter = single_roll_parameter()
    parameter['dt'] = 0.001
    
    pbar = tqdm.tqdm(desc = 'RME:')    
    RME = RollManeuverExperiment()        
    
    worm = init_worm(parameter)

    CS = RME.single_roll(worm, parameter)
    
    if True:    
        plot_strains(CS.to_numpy())
        plt.show()

    FS, CS, _, e = simulate_experiment(worm, parameter, CS, pbar = pbar)

    if e is not None:
        raise e

    print(f'Picard iteration converged at every time step: {np.all(FS.pic)}')
    
    plot_roll_maneuver(FS, CS, parameter)
         
#------------------------------------------------------------------------------ 
# Plotting
                    
def plot_roll_maneuver(FS, CS, parameter):
    
    plot_CS_vs_FS(CS, FS, T = parameter['T'])
    
    alpha = FS.theta[:, 0, :]
    alpha = alpha % (2 * np.pi) - np.pi    
    alpha = 180*alpha / np.pi
    
    k3 = FS.Omega[:, 2, :]
    w3 = FS.w[:, 2, :]
    
    plot_multiple_scalar_fields([alpha, k3, w3],
        titles = [r'$\alpha$', r'$\omega_3$', r'$\kappa_3$'],    
        cmaps = [plt.cm.seismic, plt.cm.seismic, plt.cm.PRGn]
        )
    
    plt.show()
            
    return
                                      
if __name__ == "__main__":
    
    test_continuous_roll_maneuver(plot_control = False,
         make_video=False, active_clip=False, plot=False)    
            
    #test_single_roll_maneuver()
    
    print('Finished')

