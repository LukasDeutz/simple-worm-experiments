
# Build-in imports
from os.path import isfile
import pickle
from multiprocessing import Pool
from pathlib import Path

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Local imports
from simple_worm.plot3d import generate_interactive_scatter_clip
from simple_worm.plot3d_cosserat import plot_controls_CS_vs_FS, plot_strains 
from simple_worm.plot3d_cosserat import plot_single_strain_vs_control, plot_single_strain

from simple_worm_experiments.util import get_solver 
from simple_worm_experiments.roll.roll import RollManeuverExperiment
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm_experiments.worm_videos import WormStudio

from simple_worm_experiments.experiment import simulate_experiment, init_worm
from numpy import dtype

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
    parameter['G'] = parameter['E'] / (2 * (1 + 0.5))
    parameter['eta'] = 1e-2 * parameter['E'] 
    parameter['nu'] = 1e-2 * parameter['G']
        
    # Muscle parameter        
    parameter['smo'] = False
    parameter['Ds'] = 1.0/150
    parameter['s0'] = 0.05
    
    parameter['mts'] = True
    parameter['tau_on'] = 0.01
    parameter['tau_off'] = 0.01    
    parameter['Dt_on'] = 3*parameter['tau_on']
    parameter['Dt_off'] = 3*parameter['tau_off']
        
    return parameter

def repetitious_roll_parameter():

    parameter = base_parameter()
    
    # Kinematic parameter
    parameter['A_dv'] = 3.0
    parameter['f_dv'] = 5.0    

    parameter['A_lr'] = 3.0
    parameter['f_lr'] = 5.0    
    parameter['phi'] = np.pi/2

    parameter['Ds_h'] = 1.0/16
    parameter['s0_h'] = 0.75
    
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
    parameter['Dt_roll'] = 0.75 / parameter['f_dv']

    # Muscle parmaeter    
    parameter['Ds_h'] = 1.0/16
    parameter['s0_h'] = 0.75
    
    return parameter
           
def test_continuous_roll_maneuver(make_video = True, 
        active_clip = True, 
        plot = True):
        
    parameter = repetitious_roll_parameter()
    parameter['T'] = 2.5
    parameter['dt'] = 0.01
    parameter['A_dv'] = 6.0
    
    pbar = tqdm.tqdm(desc = 'RME:')    
    RME = RollManeuverExperiment()        
    
    worm = init_worm(parameter)

    CS = RME.continuous_roll(worm, parameter)
        
    if False:    
        plot_strains(CS.to_numpy())
        plt.show()

    FS, CS, _, e = simulate_experiment(worm, parameter, CS, pbar = pbar)

    if e is not None:
        raise e

    print(f'Picard iteration converged at every time step: {np.all(FS.pic)}')
        
    
    if make_video:        
        WS = WormStudio(FS)        
        output_path = Path(f'../videos/continuous_A_{parameter["A_dv"]}_f_{parameter["f_dv"]}_roll')
        WS.generate_clip(output_path, add_trajectory = False, add_frame_vectors = False)        
    if active_clip:
        generate_interactive_scatter_clip(FS, 500, n_arrows=25)    
    if plot:
        plot_roll_maneuver(FS, CS, parameter)
    
    
    
            
    return

def comp_roll_frequency(FS, parameter):

    # Compute roll frequency
    s_arr = np.linspace(0, 1, parameter['N'])    
    s0_h = parameter['s0_h']
    s_mask = s_arr >= (1 - s0_h)
        
    d2 = FS.e2
    
    d1_ref = FS.e1[0, :, 0] 
    d2_ref = FS.e2[0, :, 0] 
    d3_ref = FS.e3[0, :, 0] 
    
    d123_ref = np.vstack((d1_ref, d2_ref, d3_ref))
        
    f_avg, f_std, phi = EPP.comp_roll_frequency(d2, 
        FS.times, 
        d123_ref, 
        s_mask = s_mask, 
        Dt = 0.25)
    
    return f_avg, f_std, phi
    

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
        
             
def plot_roll_maneuver(FS, CS, parameter):
            
    # Plot controls and real strains
    #plot_controls_CS_vs_FS(CS, FS, parameter['dt'])
        
    f_avg, f_std, phi = comp_roll_frequency(FS, parameter)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(FS.times, phi)
        
    plt.show()
    
    # print(f'f_head: {parameter["f_dv"]}')   
    # print(f'f_avg: {f_avg:.2f}')
    # print(f'f_std: {f_std:.3f}')
        
    return
                                      
if __name__ == "__main__":
    
    test_continuous_roll_maneuver(make_video=False, active_clip=False, plot=True)    
    #test_single_roll_maneuver()
    print('Finished')

