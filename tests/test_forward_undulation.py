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

from simple_worm_experiments.util import default_parameter, get_solver, comp_angle_of_attack
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

    gmo_t = True
    tau_on = 0.05
    Dt_on = 3*tau_on
        
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

    parameter['gmo_t'] = gmo_t
    parameter['tau_on'] = tau_on
    parameter['Dt_on'] = Dt_on
        
    return parameter

def run_foward_undulation(parameter):

    pbar = tqdm.tqdm(desc = 'Test Undulation:')        
    solver = get_solver(parameter)    
    FU = ForwardUndulationExperiment(parameter['N'], parameter['dt'], solver = solver, quiet=True)
    
    FS, CS, MP, e = FU.simulate_undulation(parameter, pbar = pbar)
            
    if e is not None:
        raise e
            
    return FS, CS, MP
                

def test_forward_undulation():
    
    parameter = get_test_parameter()    
    FS, CS, _ = run_foward_undulation(parameter)

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

def test_forward_work(show = False):
    
    parameter = get_test_parameter()        
    parameter['T'] = 2.5
    parameter['dt'] = 0.001
    parameter['N_report'] = 100
    parameter['dt_report'] = 0.01    
    parameter['fdo'] = {1:1, 2:1}
    parameter['pi'] = False
    
    parameter['eta'] = 1e-3*parameter['E'] 
    parameter['nu'] = 1e-3*parameter['G']
    
    FS, CS, MP = run_foward_undulation(parameter)
                                            
    dot_w_F_lin = FS.dot_w_F_lin
    dot_w_F_rot = FS.dot_w_F_rot
    
    dot_W_F_lin = FS.dot_W_F_lin
    dot_W_F_rot = FS.dot_W_F_rot
    
    dot_w_M_lin = FS.dot_w_M_lin
    dot_w_M_rot = FS.dot_w_M_rot
    
    dot_W_M_lin = FS.dot_W_M_lin
    dot_W_M_rot = FS.dot_W_M_rot
    
    # Check if fenics and numerical integration are equal
    # up to some tolerance
    ds = 1.0 / (parameter['N'] - 1)        
    dot_W_F_lin_np = trapezoid(dot_w_F_lin, dx = ds, axis = 1)
    dot_W_F_rot_np = trapezoid(dot_w_F_rot, dx = ds, axis = 1)

    dot_W_M_lin_np = trapezoid(dot_w_M_lin, dx = ds, axis = 1)
    dot_W_M_rot_np = trapezoid(dot_w_M_rot, dx = ds, axis = 1)
    
    assert np.allclose(dot_W_F_lin, dot_W_F_lin_np)
    assert np.allclose(dot_W_F_rot_np, dot_W_F_rot_np)
    
    assert np.allclose(dot_W_M_lin, dot_W_M_lin_np)
    assert np.allclose(dot_W_M_rot_np, dot_W_M_rot_np)
    
    t = FS.times
    T_undu = 1.0/parameter['f']        
    idx = t >= T_undu
    T = parameter['T']
    
    dot_V = FS.V_dot_k + FS.V_dot_sig
    dot_D_B = FS.D_k + FS.D_sig

    assert np.all(dot_D_B)
    
    dot_W_F = dot_W_F_lin + dot_W_F_rot
    dot_W_M = dot_W_M_lin + dot_W_M_rot
    
    assert np.all(dot_W_F <= 0)
    
        
    dot_W_F_avg = np.mean(dot_W_F[idx])
    dot_W_M_avg = np.mean(dot_W_M[idx])
    
    # estimate order of magnitude
    w = FS.w[idx, :, :]
    u = FS.x_t[idx, :, :]    
    k = FS.Omega[idx, :, :]
    sig = FS.sigma[idx, :, :]
    f = FS.f[idx, :, :]

    dt = parameter['dt']
    sig_t = np.gradient(sig, dt, axis = 0)
    k_t = np.gradient(k, dt, axis = 0)
                                    
    w_avg = np.mean(np.linalg.norm(w, axis = 1))
    u_avg = np.mean(np.linalg.norm(u, axis = 1))
    k_avg = np.mean(np.linalg.norm(k, axis = 1))
    sig_avg = np.mean(np.linalg.norm(sig, axis = 1))

    k_t_avg = np.mean(np.linalg.norm(k_t, axis = 1))
    sig_t_avg = np.mean(np.linalg.norm(sig_t, axis = 1))
        
    f_avg = np.mean(np.linalg.norm(f, axis = 1))

    print(f'w: {w_avg}')
    print(f'u: {u_avg}')
    print(f'k: {k_avg}')
    print(f'sig: {sig_avg}')
    print(f'k_t: {k_t_avg}')
    print(f'sig_t: {sig_t_avg}')    
    print(f'f: {f_avg}')
    print(f'B: {MP.B}')
    print(f'S: {MP.S}')
    print(f'S: {MP.S}')
    
    gs = plt.GridSpec(2,1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
            
    ax0.plot(t, dot_W_F, c = 'b', label = r'$\dot{D}_F$')
    ax0.plot(t, dot_D_B, c = 'r', label = r'$\dot{D}_B$')
    ax0.plot(t, -dot_W_M, c = 'orange', label = r'$\dot{W}_M$')
    ax0.plot(t, dot_V, c = 'g', label = '$\dot{V}$')
    
    ax0.legend()
    
    dot_D = dot_W_F + dot_D_B
        
    ax1.plot(t, dot_D - dot_V, c = 'k', label = '$\dot{D}$')
    ax1.plot(t, -dot_W_M, c = 'orange', ls = '--', label = '$\dot{W}_M$')
    
    ax1.legend()
            
    # plt.plot([0, T], [dot_W__avg, dot_W_lin_avg])
        
    if show:
        plt.show()
    else:    
        plt.savefig('undulation_work.png')
                                     
    return
          
def test_forward_undulation_angle_of_attack(show = True):

    parameter = get_test_parameter()        
    parameter['T'] = 1.0
    parameter['N_report'] = 100
    parameter['dt_report'] = 0.01
    parameter['fdo'] = {1:1, 2:1}
    parameter['pi'] = False

    FS, CS = run_foward_undulation(parameter)
                    
    x = FS.x
    t = FS.times
    T_undu = 1.0 / parameter['f']
                        
    theta, avg_theta, t_avg_theta = comp_angle_of_attack(x, t, T_undu)    
    
    ax = plt.subplot(111)
    
    t_avg_theta = 360 * t_avg_theta / (2 * np.pi)
    
    ax.plot(t, 360 * avg_theta / (2 * np.pi), c = 'k')
    ax.plot([t[0], t[-1]], [t_avg_theta, t_avg_theta], ls = '--', c='k')
    ax.set_xlabel('$t$', fontsize = 20)
    ax.set_ylabel('$\theta$', fontsize = 20)
        
    if show:
        plt.show()
        
    plt.savefig('undulation_work.png')

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
    #test_forward_undulation_work(show=True)    
    #test_forward_undulation_angle_of_attack(show = True)
    test_forward_work(show = True)
    
    print('Finished')

