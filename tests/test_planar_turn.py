# Build-in imports
from os.path import isfile
import pickle
from multiprocessing import Pool
from math import ceil

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Local imports
from simple_worm.plot3d import generate_interactive_scatter_clip
from simple_worm.plot3d_cosserat import plot_controls_CS_vs_FS, plot_single_strain_vs_control

from simple_worm_experiments.util import default_parameter, get_solver, comp_minimum_distant, comp_average_distant
from simple_worm_experiments.planar_turn.planar_turn import PlanarTurnExperiment, wrap_simulate_planar_turn
from simple_worm_experiments.planar_turn.planar_turn_util import compute_turning_angle
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
    A0 = 4.0
    lam0 = 1.5
    f0 = 2.0    
    
    A1 = 8.0
    lam1 = 1.5
    f1 = 2.0    
       
    T0 = 1.0 / f0
    T1 = 1.0 / f1
       
    T = round(5.0*T0, 1)
    t0 = 2*T0
    Delta_t = 0.5 * T1
        
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

def sim_planar_turn(parameter):
    
    pbar = tqdm.tqdm(desc = 'PTE:')    
    PTE = PlanarTurnExperiment(parameter['N'], parameter['dt'], solver = get_solver(parameter), quiet = True)        
    FS, CS, _, e = PTE.simulate_planar_turn(parameter, pbar = pbar)
    
    return FS, CS, _, e
           
def test_planar_turn():
        
    parameter = get_base_parameter()

    parameter['N'] = 100
    parameter['dt'] = 0.01        
    parameter['pi'] = False
    
    FS, CS, _, e = sim_planar_turn(parameter)
    
    if e is not None:
        raise e

    phi = compute_turning_angle(FS.x, FS.times, parameter)
    phi = 180 * phi / np.pi

    print(f'Finished simulation: Picard iteration convergence rate: {np.sum(FS.pic) / len(FS.pic)}')
    print(f'Turning angle: {round(phi,1)}')
    
    if False:
                
        CS.to_numpy()
        generate_interactive_scatter_clip(FS, 500, perspective = 'xy', n_arrows = 50) # n_arrows= 65                               
    
        plot_controls_CS_vs_FS(CS.to_numpy(), FS, parameter['dt'])        
        k0 = CS.Omega[:, 0, :]
        k = FS.Omega[:, 0, :]        
        
        plot_single_strain_vs_control(k0, k, dt = parameter['dt'], titles = [r'$\kappa_{2,0}$', r'$\kappa_{2}$'], cbar_format='%.1f', cmap = plt.get_cmap('plasma'))        
        plot_trajectory(FS, parameter)
        plt.show()    

def test_distant_matrix():
    
    parameter = get_base_parameter()

    N = 100
    dt = 0.01
    parameter['T'] = 2.5
    parameter['N'] = N
    parameter['dt'] = dt         
    parameter['pi'] = False
    
    L0 = parameter['L0']     
    r_max = parameter['r_max'] 
    
    r_max = r_max / L0        
    ds = 1.0 / (N - 1)
    
    # Here, we use an arc-length distance threshold. 
    # For every point, all neighbouring centreline points which 
    # have a smaller distance than the threshold in 
    # the natural configuration are excluded.      
    ds_th = 4*r_max    
    n_skip = ceil(ds_th / ds)
                        
    FS, CS, _, e = sim_planar_turn(parameter)
    
    if e is not None:
        raise e


    x_min_dist, x_min_dist_arr = comp_minimum_distant(FS.x, n_skip)
    x_avg_dist, x_avg_dist_arr = comp_average_distant(FS.x)

    gs = plt.GridSpec(2, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    # Plot minimum distance between all pairs of points along the body             
    ax0.plot(FS.times, x_min_dist)
    
    # Minimum allowed distance, for smaller distances worm body starts to intersect ifself
    d_min = 2*r_max
    ax0.plot([FS.times[0], FS.times[-1]], [d_min, d_min], '--', c='k')
    # Plot average distance, for smaller average distance 
    # interactions should become more important and the 
    # difference between rft and sbt more significant
    ax1.plot(FS.times, x_avg_dist)
    
    # Highlight time window of turn maneuver
    t0 = parameter['t0']    
    t1 = t0 + parameter['Delta_t']    
    ymin, ymax = plt.ylim()
    y = np.linspace(0, ymax, int(1e3), endpoint=True)
    ax0.fill_betweenx(y, t0, t1, color = 'r', alpha = 0.3)
     
    ax1.fill_betweenx(y, t0, t1, color = 'r', alpha = 0.3)
                
    plt.show()
    
    return
                                  
if __name__ == "__main__":
    
    #test_planar_turn()    
    test_distant_matrix()
    
    print('Finished')

