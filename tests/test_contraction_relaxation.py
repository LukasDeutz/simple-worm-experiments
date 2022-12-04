# Build-in imports
from os.path import isfile

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.integrate import trapezoid, cumulative_trapezoid

# Local imports
from simple_worm.plot3d import generate_interactive_scatter_clip
from simple_worm.plot3d_cosserat import plot_controls_CS_vs_FS, plot_single_strain_vs_control, plot_single_strain

from simple_worm_experiments.util import default_parameter, get_solver, comp_angle_of_attack, frame_trafo
from simple_worm_experiments.contraction_relaxation.contraction_relaxation import ContractionRelaxationExperiment
from scipy.integrate._quadrature import cumtrapz
from sympy.parsing.tests import test_latex_deps

def get_test_parameter():

    # Simulation parameter
    N = 100
    dt = 0.01
    T = 1.0
    T0 = 0.5 
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
    tau_off = 0.05 
    Dt_off = 3*tau_off
        
    parameter = {}
              
    parameter['N']  = N
    parameter['dt'] = dt
    parameter['T'] = T
    parameter['T0'] = T0
    
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
    parameter['tau_off'] = tau_on 
    parameter['Dt_off'] = Dt_off 

    return parameter

#------------------------------------------------------------------------------ 
#

def run_contraction_relaxation(parameter):

    pbar = tqdm.tqdm(desc = 'Test CRE:')        
    solver = get_solver(parameter)    
    CRE = ContractionRelaxationExperiment(parameter['N'], parameter['dt'], solver = solver, quiet=True)
    
    FS, CS, _, e = CRE.simulate_contraction_relaxation(parameter, pbar = pbar)
            
    if e is not None:
        raise e
            
    return FS, CS


#------------------------------------------------------------------------------ 
# Plotting

def plot_sigmoid():
    '''
    Estimate timescale for muscle on an off switch. 
    '''
    
    # Set time duration for muscle on/off switch
    Dt = 0.1       
        
    # After Dt, we want want the muscles to have 
    # reached 1-eps of their full strength    
    eps_arr = np.array([1e-1, 1e-2, 1e-3, 1e-4])
        
    t_arr = np.linspace(0, 4*Dt, int(1e3))
    
    # Centre sigmoid around 3*Dt
    t0 = 2*Dt
    
    ax = plt.subplot(111)
    
    # Compute time muscle scale tau_M for different eps           
    tau_M_arr = np.zeros_like(eps_arr)
                
    for i, eps in enumerate(eps_arr):
    
        tau_M = Dt / (2*(np.log((1-eps)/eps)))         
        s_arr = 1.0 / (1.0 + np.exp(-(t_arr -t0) / tau_M))
    
        ax.plot(t_arr, s_arr, label = f'$\\tau={tau_M:.3f}$')
        tau_M_arr[i] = tau_M
                            
    ax.plot([2.5*Dt, 2.5*Dt], [0, 1], ls='--', c='k')
        
    plt.legend()
    plt.show()
    
    return

def plot_experiment_clib():
        
    parameter = get_test_parameter()    
            
    parameter['T'] = 6.0
    parameter['T0'] = 3.0
    parameter['dt'] = 0.01
                    
    # Make fluid very viscous
    parameter['mu'] = 1e-0
    
    # Set body viscosity to zero
    parameter['eta'] = 1e-3 * parameter['E'] 
    parameter['nu'] = 1e-3 * parameter['G']
        
    parameter['k0'] = np.array([np.pi, 0, 0])
    parameter['sig0'] = np.array([0, 0, 0])
    parameter['pi'] = False
        
    FS, CS = run_contraction_relaxation(parameter)
    
    generate_interactive_scatter_clip(FS, 500, n_arrows = 25, perspective = 'xy')
            
    plt.show()

    return

def plot_all_energies():
    
    parameter = get_test_parameter()    

    # Make fluid very viscous
    parameter['mu'] = 1e-0
        
    # Increase simulation duration
    parameter['T'] = 4.0
    parameter['T0'] = 2.0
    parameter['dt'] = dt
                        
    # Set body viscosities
    parameter['eta'] = 1e-3 * parameter['E'] 
    parameter['nu'] = 1e-3 * parameter['G']
        
    parameter['k0'] = np.array([np.pi, 0, 0])
    parameter['sig0'] = np.array([0, 0, 0])
    parameter['pi'] = False
    
    FS, _ = run_contraction_relaxation(parameter)

    dot_D_k = FS.D_k
    dot_D_sig = FS.D_sig

    D_k = cumulative_trapezoid(dot_D_k, dx=dt, initial=0)
    D_sig = cumulative_trapezoid(dot_D_sig, dx=dt, initial=0)
                            
    dot_W_F_lin = FS.dot_W_F_lin
    dot_W_F_rot = FS.dot_W_F_rot
            
    W_fluid_lin = cumulative_trapezoid(dot_W_F_lin, dx=dt, initial=0)
    W_fluid_rot = cumulative_trapezoid(dot_W_F_rot, dx=dt, initial=0)

    dot_W_M_lin = FS.dot_W_M_lin
    dot_W_M_rot = FS.dot_W_M_rot
    
    W_M_lin = cumulative_trapezoid(dot_W_M_lin, dx=dt, initial=0)
    W_M_rot = cumulative_trapezoid(dot_W_M_rot, dx=dt, initial=0)
    
    gs = plt.GridSpec(4, 2)
    ax00 = plt.subplot(gs[0, 0])    
    ax10 = plt.subplot(gs[1, 0])
    ax20 = plt.subplot(gs[2, 0])
    ax30 = plt.subplot(gs[3, 0])
        
    ax01 = plt.subplot(gs[0, 1])
    ax11 = plt.subplot(gs[1, 1])
    ax21 = plt.subplot(gs[2, 1])
    ax31 = plt.subplot(gs[3, 1])

    ax00_twin = ax00.twinx()        
    ax00_twin.plot(t, dot_V_k, label = '$\dot{V}_{\kappa}$', c = 'g')
    ax00.plot(t, V_k, ls = '--', label = '$V_{\kappa}$', c = 'g')        
    
    ax01_twin = ax01.twinx()            
    ax01_twin.plot(t, dot_V_sig, label = '$\dot{V}_{\sigma}$', c = 'orange')
    ax01.plot(t, V_sig, ls = '--', label = '$V_{\sigma}$', c = 'orange')

    ax10_twin = ax10.twinx()
    ax10_twin.plot(t, dot_D_k, c = 'm', label = '$\dot{D}_{\kappa}$')
    ax10.plot(t, D_k, ls = '--', c = 'm', label = '$D_{\kappa}$')

    ax11_twin = ax11.twinx()    
    ax11_twin.plot(t, dot_D_sig, c = 'c', label = '$\dot{D}_{\sigma}$')    
    ax11.plot(t, D_sig, ls = '--', c ='c', label = '$D_{\sigma}$')

    ax20_twin = ax20.twinx()    
    ax20_twin.plot(t, dot_W_F_lin, c = 'b', label = '$\dot{W}_{\text{Fluid}}$')    
    ax20.plot(t, W_fluid_lin, ls = '--', c ='b', label = '$W_{\text{Fluid}}$')
    
    ax21_twin = ax21.twinx()        
    ax21_twin.plot(t, dot_W_F_rot, c = 'r', label = '$\dot{W}_{\text{Fluid}}$')        
    ax21.plot(t, W_fluid_rot, ls = '--', c ='r', label = '$W_{\text{Fluid}}$')
                                    
    ax21_twin = ax21.twinx()        
    ax21_twin.plot(t, dot_W_F_rot, c = 'r', label = '$\dot{D}_{\sigma}$')        
    ax21.plot(t, W_fluid_rot, ls = '--', c ='r', label = '$D_{\sigma}$')

    ax30_twin = ax30.twinx()    
    ax30_twin.plot(t, dot_W_M_lin, c = 'y', label = '$\dot{W}_{\text{Muscle}}$')    
    ax30.plot(t, W_M_lin, ls = '--', c ='y', label = '$W_{\text{Muscle}}$')
    
    ax31_twin = ax31.twinx()    
    ax31_twin.plot(t, dot_W_M_rot, c = 'k', label = '$\dot{W}_{\text{Muscle}}$')    
    ax31.plot(t, W_M_rot, ls = '--', c ='k', label = '$W_{\text{Muscle}}$')

    plt.show()
    
    return
    
#------------------------------------------------------------------------------ 
# Testing
      
def test_energy_rate_balance():
    
    parameter = get_test_parameter()    

    dt = 0.001

    parameter['T'] = 4.0
    parameter['T0'] = 2.0
    parameter['dt'] = dt
        
    # Muscle time scale    
    tau_on = 0.05    
    parameter['tau_on']  = tau_on
    parameter['Dt_on'] = 5*tau_on
                
    tau_off = 0.05
                    
    parameter['tau_off'] = tau_off
    parameter['Dt_off'] = 5*tau_off
        
    # Make fluid very viscous
    parameter['mu'] = 1e-0
        
    # Set body viscosities
    parameter['eta'] = 1e-2 * parameter['E'] 
    parameter['nu']  = 1e-2 * parameter['G']
        
    parameter['k0'] = np.array([np.pi, 0, 0])
    parameter['sig0'] = np.array([0, 0, 0])
    parameter['pi'] = False
                
    FS, CS = run_contraction_relaxation(parameter)    
    
    t = FS.times
    
    dot_V_k = FS.V_dot_k
    dot_V_sig = FS.V_dot_sig        
    dot_D_k = FS.D_k
    dot_D_sig = FS.D_sig
            
    dot_E = - dot_V_k - dot_V_sig + dot_D_k + dot_D_sig + FS.dot_W_F_lin + FS.dot_W_F_rot                                                 
    dot_W_M = FS.dot_W_M_lin + FS.dot_W_M_rot
                            
    tol = 1e-2
    
    gs = plt.GridSpec(3,1)
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
              
    ax0.semiloy(t, np.abs(dot_E))
    ax0.semiloy(t, np.abs(dot_W_M))
        
    plt.show()
                                    
    assert np.allclose(err, 0.0 , tol)
                    
    return
    
      
                
def test_body_dissipation():

    parameter = get_test_parameter()    

    T = 2.0
    T0 = 1.0
    dt = 0.01

    parameter['T'] = 5.0
    parameter['T0'] = 2.5
    parameter['dt'] = 0.01
    
    # Make fluid very viscous
    parameter['mu'] = 1e-3
        
    # Set body viscosities
    parameter['eta'] = 1e0 * parameter['E'] 
    parameter['nu'] = 1e0 * parameter['G']
        
    parameter['k0'] = np.array([np.pi, 0, 0])
    parameter['sig0'] = np.array([0, 0, 0])
    parameter['pi'] = False
        
    FS, CS = run_contraction_relaxation(parameter)
    
    t = FS.times
    D_k = FS.D_k
    D_sig = FS.D_sig

    ax = plt.subplot(111)
    
    ax.plot(t, D_k, label = '$D_{\kappa}$')
    ax.plot(t, D_sig, label = '$D_{\sigma}$')
    ax.legend()
    
    plt.show()
    
    return

def init_controls():
    
    pass
    
                      
    # dot_W = dot_W - dot_W_M_rot 
    #
    # W = cumulative_trapezoid(dot_W, dx = dt, initial = 0)
    #
    # ax2.plot(t, W, c = 'k')
    # ax2.plot([t[0], t[-1]], [0, 0], c='k', ls = '--')
    #
    # plt.show()
        
    return
                                                  
if __name__ == "__main__":
        
    #plot_contraction_relaxation()    
    #test_body_dissipation()    
    #plot_torque_balance()
    
    #plot_sigmoid()    
    
    test_energy_rate_balance()
        
    print('Finished')




# def plot_torque_balance():
#
#     parameter = get_test_parameter()    
#
#     T = 0.1
#     T0 = 1.0
#     dt = 0.01
#
#     parameter['T'] = 2.0
#     parameter['T0'] = 1.0
#     parameter['dt'] = 0.01
#
#     # Make fluid very viscous
#     parameter['mu'] = 1e-0
#
#     # Set body viscosities
#     parameter['eta'] = 1e-2 * parameter['E'] 
#     parameter['nu'] = 1e-2 * parameter['G']
#
#     parameter['k0'] = np.array([np.pi, 0, 0])
#     parameter['sig0'] = np.array([0, 0, 0])
#     parameter['pi'] = False
#
#     FS, CS = run_contraction_relaxation(parameter)
#
#     r = FS.x
#     ds = 1.0 / (parameter['N'] - 1)
#
#     dr_ds = np.gradient(r, ds, axis = 2)
#
#     l = frame_trafo(FS.l, FS, trafo = 'b2l') 
#     m = frame_trafo(FS.m, FS, trafo = 'b2l') 
#     n = frame_trafo(FS.n, FS, trafo = 'b2l')
#
#     dr_ds_arr = dr_ds[-1, :, :]
#     l = l[-1, :, :]
#
#     m = m[-1, :, :]
#     dm_ds = np.gradient(m, ds, axis = 1)
#     n = n[-1, :, :]
#
#     gs = plt.GridSpec(3,1)
#     ax0 = plt.subplot(gs[0])
#     ax1 = plt.subplot(gs[1])
#     ax2 = plt.subplot(gs[2])
#
#     dr_ds_cross_n = np.zeros_like(n)
#
#     for i, (dr_ds, n) in enumerate(zip(dr_ds_arr.T, n.T)):
#
#         dr_ds_cross_n[:,i] = np.cross(dr_ds, n)    
#
#     ax0.plot(l[0, :])
#     ax0.plot(dr_ds_cross_n[0, :])
#     ax0.plot(dm_ds[0, :])
#
#     ax1.plot(l[1, :], label = 'l')
#     ax1.plot(dr_ds_cross_n[1, :], label = 'cross')
#     ax1.plot(dm_ds[1, :], label = 'm')
#     ax1.legend()
#
#     ax2.plot(l[2, :])
#     ax2.plot(dr_ds_cross_n[2, :])
#     ax2.plot(dm_ds[2, :])
#
#     plt.show()
#
#     return

