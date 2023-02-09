# Third party imports
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.integrate import trapezoid

# Local imports
from simple_worm.plot3d import generate_interactive_scatter_clip
from simple_worm.plot3d_cosserat import plot_CS_vs_FS, plot_S

from simple_worm_experiments.experiment import simulate_experiment, init_worm
from simple_worm_experiments.forward_undulation.undulation import UndulationExperiment
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm_experiments.worm_studio import WormStudio

#------------------------------------------------------------------------------ 
# Parameter

def base_parameter():
                    
    parameter = {}
          
    # Simulation parameter                      
    parameter['N']  = 100
    parameter['dt'] = 0.01
    parameter['T'] = 2.5
    parameter['N_report'] = None
    parameter['dt_report'] = None

    # Solver parameter        
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

    # Finite muscle timescale        
    parameter['fmts'] = True    
    parameter['tau_on'] = 0.01
    parameter['Dt_on'] = 3*parameter['tau_on']
                
    return parameter

def undulation_parameter():

    parameter = base_parameter()

    # Kinematic parameter
    parameter['A'] = 5.0
    parameter['lam'] = 1.0
    parameter['f'] = 2.0
    
    # Gradual muscle onset at head and tale
    parameter['gmo'] = True
    parameter['Ds_h'] = 0.01
    parameter['s0_h'] = 3*parameter['Ds_h']    
    parameter['Ds_t'] = 0.01
    parameter['s0_t'] = 1 - 3*parameter['Ds_t']

    return parameter

#------------------------------------------------------------------------------ 
# "Test" scripts      
                
def test_forward_undulation(
        plot_control = False,
        make_video = False, 
        active_clip = True, 
        plot = True):
        
    parameter = undulation_parameter()    
    parameter['T'] = 2.5
    
    
    worm = init_worm(parameter)

    CS = UndulationExperiment.sinusoidal_traveling_wave_control_sequence(
            worm, parameter)
        
    if plot_control:
        plot_S(CS.to_numpy(), dt = parameter['dt'])
        plt.show()
        
    FS, CS, _, e = simulate_experiment(worm, 
        parameter, CS, pbar = tqdm.tqdm(desc = 'UE:'))
            
    if e is not None:
        raise e

    print(f'Picard iteration converged at every time step: {np.all(FS.pic)}')
                        
    if make_video:        
        WS = WormStudio(FS)        
        output_path = Path(
            f'../videos/undulation_A={parameter["A"]}_lam={parameter["lam"]}_f={parameter["f"]}')
        WS.generate_clip(output_path, 
            add_trajectory = False, 
            add_frame_vectors = True,
            draw_e3 = False,
            n_arrows = 0.2)        
    if active_clip:
        generate_interactive_scatter_clip(FS, 500, n_arrows=25)    
    if plot:
        plot_undulation(FS, CS, parameter)
            
    return

#------------------------------------------------------------------------------ 
# Plotting

def plot_undulation(FS, CS, parameter):
    
    plot_CS_vs_FS(CS, FS, T = parameter['T'])    
    
    
    # Plot COM trajectory and velocity    
    X = FS.x    
    t = FS.times
    
    x_head = X[:, :, 0]
    x_tale = X[:, :, -1]
    x_mid = X[:, :, int(0.5*X.shape[2])]
     
    x_com, v_com, _ = EPP.comp_com(X, parameter['dt'])
    U = EPP.comp_mean_com_velocity(X, t, 1.0/parameter['f'])
    
    # Plot Head/midpoint/tale trajectory    
    gs = plt.GridSpec(2, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    ax0.plot(x_com[:, 1], x_com[:, 2], ls = '-', c = 'k')
    ax0.plot(x_mid[:, 1], x_mid[:, 2], ls = '--', c = 'b')        
    ax0.plot(x_mid[:, 1], x_mid[:, 2], ls = '--', c = 'b')
    
    ax1.plot([t[0], t[-1]], [U, U], ls = '-', c = 'k')
    ax1.plot(t, v_com, ls = '--', c = 'b')
    
    plt.show()
    
    return
    
    # x_tale = X[-1, :, int(0.5*X.shape[2])]
    # x_head = X[0, :, :]
    # t = FS.times
    # theta, avg_theta, t_avg_theta = comp_angle_of_attack(
    #     FS.x, t, 1/parameter['f'])    
    #
    # fig = plt.figure()
    #
    # ax.plot(t, 360 * avg_theta / (2 * np.pi), c = 'k')
    # ax.plot([t[0], t[-1]], [t_avg_theta, t_avg_theta], ls = '--', c='k')
    # ax.set_xlabel('$t$', fontsize = 20)
    # ax.set_ylabel('$\theta$', fontsize = 20)
    #
    # plt.show()
    
def test_forward_work(show = False):
    
    parameter = get_test_parameter()        
    parameter['T'] = 2.5
    parameter['dt'] = 0.01
    parameter['N_report'] = 100
    parameter['dt_report'] = 0.01    
    parameter['fdo'] = {1:1, 2:1}
    parameter['pi'] = False
    
    parameter['eta'] = 1e-2*parameter['E'] 
    parameter['nu'] = 1e-2*parameter['G']
    
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

                                                 
if __name__ == "__main__":
    
    test_forward_undulation(plot_control = False,
        make_video = False, active_clip = False, plot = False)
    #test_forward_undulation_work(show=True)    
    #test_forward_work(show = True)
    
    print('Finished')

