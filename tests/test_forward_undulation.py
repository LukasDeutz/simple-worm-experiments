# Build-in imports
from os.path import isfile

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.integrate import trapezoid

# Local imports
from simple_worm.plot3d import generate_interactive_scatter_clip
from simple_worm.plot3d_cosserat import plot_controls_CS_vs_FS, plot_single_strain_vs_control

from simple_worm_experiments.util import default_parameter, get_solver
from simple_worm_experiments.forward_undulation.undulation import ForwardUndulationExperiment
from simple_worm_experiments.forward_undulation.plot_undulation import plot_trajectory

def get_test_parameter():

    # Simulation parameter
    N = 100
    dt = 0.01
    T = 2.5
    N_report = None
    dt_report = None
    
    # Model parameter
    external_force = ['rft']
    rft = 'Whang'
    use_inertia = False
    
    # Solver parameter
    pi_alpha0 = 0.9
    pi_maxiter = 1000
    fdo = {1: 2, 2:2}
        
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
    A = 4.0
    lam = 1.5
    f = 2.0    
    smo = True
    a_smo = 150
    du_smo = 0.05
        
    parameter = {}
          
    
    parameter['N']  = N
    parameter['dt'] = dt
    parameter['T'] = T
    parameter['N_report'] = N_report
    parameter['dt_report'] = dt_report
        
    parameter['pi_alpha0'] = pi_alpha0            
    parameter['pi_maxiter'] = pi_maxiter             
    parameter['fdo'] = fdo
                
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
        
    parameter['A'] = A
    parameter['lam'] = lam
    parameter['f'] = f    
    parameter['smo'] = smo
    parameter['a_smo'] = a_smo
    parameter['du_smo'] = du_smo

    return parameter

def run_foward_undulation(parameter):

    pbar = tqdm.tqdm(desc = 'Test Undulation:')        
    solver = get_solver(parameter)    
    FU = ForwardUndulationExperiment(parameter['N'], parameter['dt'], solver = solver, quiet=True)
    
    FS, CS, _, e = FU.simulate_undulation(parameter, pbar = pbar)
            
    if e is not None:
        raise e
            
    return FS, CS
                

def test_forward_undulation():
    
    parameter = get_test_parameter()    
    FS, CS = run_foward_undulation(parameter)

    print(f'Finished simulation, Picard iteration convergence rate is: {np.sum(FS.pic) / len(FS.pic)}')

    if False:    
        
        CS = CS.to_numpy()
        
        generate_interactive_scatter_clip(FS, 500, perspective = 'xy', n_arrows = 50) # n_arrows= 65                               
    
        plot_controls_CS_vs_FS(CS.to_numpy(), FS, parameter['dt'])    
    
        k0 = CS.Omega[:, 0, :]
        k = FS.Omega[:, 0, :]        
        
        plot_single_strain_vs_control(k0, k, dt = parameter['dt'], titles = [r'$\kappa_{2,0}$', r'$\kappa_{2}$'], cbar_format='%.1f', cmap = plt.get_cmap('plasma'))        
        plot_trajectory(FS, parameter)
        plt.show()


    return

def test_forward_undulation_work():
    
    parameter = get_test_parameter()        
    
    FS, CS = run_foward_undulation(parameter)
            
    dot_w_lin = FS.dot_w_lin
    dot_w_rot = FS.dot_w_rot
    
    dot_W_lin = FS.dot_W_lin
    dot_W_rot = FS.dot_W_rot
    
    # Check if fenics and numerical integration are equal
    # up to some tolerance
    ds = 1.0 / (parameter['N'] - 1)        
    dot_W_lin_np = trapezoid(dot_w_lin, dx = ds, axis = 1)
    dot_W_rot_np = trapezoid(dot_w_rot, dx = ds, axis = 1)
        
    assert np.allclose(dot_W_lin, dot_W_lin_np)
    assert np.allclose(dot_W_rot_np, dot_W_rot_np)
    
    t = FS.times
    T_undu = 1.0/parameter['f']        
    idx = t >= T_undu
    T = parameter['T']
    
    dot_W_lin_avg = np.mean(dot_W_lin[idx])
    dot_W_rot_avg = np.mean(dot_W_rot[idx])
    
    plt.plot(t, dot_W_lin)
    plt.plot([0, T], [dot_W_lin_avg, dot_W_lin_avg])
    plt.show()
                                     
    return
               
def test_report_lower_resolution():        

    parameter = get_test_parameter()    
    
    parameter['T'] = 0.1
    
    parameter['pi'] = False    
    parameter['N'] = 500
    parameter['dt'] = 0.001    
    parameter['N_report'] = 100
    parameter['dt_report'] = 0.01
    
    # parameter['N_report'] = 100
    # parameter['dt_report'] = 0.01
                
    FS, CS = run_foward_undulation(parameter)

    print(f'Finished simulation, Picard iteration convergence rate is: {np.sum(FS.pic) / len(FS.pic)}')

    return
                  
                                  
if __name__ == "__main__":
    
    #test_forward_undulation()
    #test_report_lower_resolution()
    
    test_forward_undulation_work()
    
    print('Finished')

