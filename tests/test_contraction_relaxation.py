# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.integrate import cumulative_trapezoid 

# Local imports
from simple_worm.plot3d import generate_interactive_scatter_clip
from simple_worm.plot3d_cosserat import plot_CS_vs_FS, plot_S

from simple_worm_experiments.experiment import simulate_experiment, init_worm
from simple_worm_experiments.model_parameter import default_model_parameter
from simple_worm_experiments.contraction_relaxation.contraction_relaxation import ContractionRelaxationExperiment
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm_experiments.worm_studio import WormStudio

#------------------------------------------------------------------------------ 
# parameter

def contraction_relaxation_parameter():
    
    param = default_model_parameter()  

    param['T'] = 6.0
    
    # Muscles switch on and off at finite timescale
    param['fmts'] = True
    param['tau_on'] = 0.03
    param['t_on'] = 5 * param['tau_on']

    param['tau_off'] = 0.03
    param['t_off'] = 3.0 

    # Gradual muscle onset at head and tale
    param['gmo'] = True
    param['Ds_h'] = 0.01
    param['s0_h'] = 3*param['Ds_h']    
    param['Ds_t'] = 0.01
    param['s0_t'] = 1 - 3*param['Ds_t']
                                
    # Make fluid very viscous
    param['mu'] = 1e-0
  
    # Constant curvature          
    param['k0'] = np.pi

    return param
           
#------------------------------------------------------------------------------ 
# "Test" scripts      
                
def test_contraction_relaxation(
        plot_control = False,
        make_video = False, 
        active_clip = True, 
        plot = True):
        
    param = contraction_relaxation_parameter()
    param['T'] = 6.0
                
    worm = init_worm(param)

    CS = ContractionRelaxationExperiment.relaxation_control_sequence(
        worm, param)
        
    if plot_control:
        plot_S(CS.to_numpy(), dt = param['dt'])
        plt.show()
        
    FS, CS, _, e = simulate_experiment(worm, 
        param, CS, pbar = tqdm.tqdm(desc = 'UE:'))
            
    if e is not None:
        raise e

    print(f'Picard iteration converged at every time step: {np.all(FS.pic)}')
                        
    if make_video:        
        WS = WormStudio(FS)        
        output_path = Path(
            f'../videos/undulation_A={param["A"]}_lam={param["lam"]}_f={param["f"]}')
        WS.generate_clip(output_path, 
            add_trajectory = False, 
            add_frame_vectors = True,
            draw_e3 = False,
            n_arrows = 0.2)        
    if active_clip:
        generate_interactive_scatter_clip(FS, 500, n_arrows=25)    
    if plot:        
        plot_all_energies(FS)
        plot_power_balance(FS)
        plt.show()
    return           

#------------------------------------------------------------------------------ 
# Plotting
            
def plot_all_energies(FS):
    
    t = FS.times
    dt = t[1] - t[0]
    powers = EPP.powers_from_FS(FS)
    energies = EPP.comp_energy_from_power(powers, dt)

    plt.figure(figsize = (2*6, 4*6))        
    gs = plt.GridSpec(4, 2)
    ax00 = plt.subplot(gs[0, 0])    
    ax10 = plt.subplot(gs[1, 0])
    ax20 = plt.subplot(gs[2, 0])
    ax30 = plt.subplot(gs[3, 0])
        
    ax01 = plt.subplot(gs[0, 1])
    ax11 = plt.subplot(gs[1, 1])
    ax21 = plt.subplot(gs[2, 1])
    ax31 = plt.subplot(gs[3, 1])

    lfz = 10

    ax00_twin = ax00.twinx()        
    ax00_twin.plot(t, powers['dot_V_k'], label = r'$\dot{V}_{\kappa}$', c = 'g')
    ax00.plot(t, energies['V_k'], ls = '--', label = r'$V_{\kappa}$', c = 'g')            
    ax00.legend(fontsize = lfz)
    
    ax01_twin = ax01.twinx()            
    ax01_twin.plot(t, powers['dot_V_sig'], label = r'$\dot{V}_{\sigma}$', c = 'orange')
    ax01.plot(t, energies['V_sig'], ls = '--', label = r'$V_{\sigma}$', c = 'orange')
    ax01.legend(fontsize = lfz)

    ax10_twin = ax10.twinx()
    ax10_twin.plot(t, powers['dot_D_k'], c = 'm', label = r'$\dot{D}_{\kappa}$')
    ax10.plot(t, energies['D_k'], ls = '--', c = 'm', label = r'$D_{\kappa}$')
    ax10.legend(fontsize = lfz)

    ax11_twin = ax11.twinx()    
    ax11_twin.plot(t, powers['dot_D_sig'], c = 'c', label = r'$\dot{D}_{\sigma}$')    
    ax11.plot(t, energies['D_sig'], ls = '--', c ='c', label = r'$D_{\sigma}$')
    ax11.legend(fontsize = lfz)

    ax20_twin = ax20.twinx()    
    ax20_twin.plot(t, powers['dot_W_F_F'], c = 'b', label = r'$\dot{W}_{\mathrm{Fluid, Force}}$')    
    ax20.plot(t, energies['W_F_F'], ls = '--', c ='b', label = r'$W_{\mathrm{Fluid, Force}}$')
    ax20.legend(fontsize = lfz)
    
    ax21_twin = ax21.twinx()        
    ax21_twin.plot(t, powers['dot_W_F_T'], c = 'r', label = r'$\dot{W}_{\mathrm{Fluid, Torque}}$')        
    ax21.plot(t, energies['W_F_T'], ls = '--', c ='r', label = r'$W_{\mathrm{Fluid, Torque}}$')
    ax21.legend(fontsize = lfz)
                            
    ax30_twin = ax30.twinx()    
    ax30_twin.plot(t, powers['dot_W_M_F'], c = 'y', label = '$\dot{W}_{\mathrm{Muscle, Force}}$')    
    ax30.plot(t, energies['W_M_F'], ls = '--', c ='y', label = '$W_{\mathrm{Muscle, Force}}$')
    ax30.legend(fontsize = lfz)
    
    ax31_twin = ax31.twinx()    
    ax31_twin.plot(t, powers['dot_W_M_T'], c = 'k', label = r'$\dot{W}_{\mathrm{Muscle, Torque}}$')    
    ax31.plot(t, energies['W_M_T'], ls = '--', c ='k', label = r'$W_{\mathrm{Muscle, Torque}}$')
    ax31.legend(fontsize = lfz)
    
    return
    
def plot_power_balance(FS):
    
    t = FS.times
    powers = EPP.powers_from_FS(FS)
          
    dot_V = powers['dot_V_k'] + powers['dot_V_sig']
    dot_D = powers['dot_D_k'] + powers['dot_D_sig']
    dot_W_F = powers['dot_W_F_F'] + powers['dot_W_F_T']    
    dot_W_M = powers['dot_W_M_F'] + powers['dot_W_M_T']
    
    dot_E_out = - dot_V + dot_D + dot_W_F  
            
    plt.figure(figsize = (10, 2*6))                                                        
    gs = plt.GridSpec(2,1)
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
              
    ax0.plot(t, dot_E_out, label = '$\dot{E}_\mathrm{out}$')
    ax0.plot(t, dot_W_M, label = '$\dot{W}_\mathrm{Muscle}$')
                                              
    ax1.semilogy(t, np.abs(dot_E_out - dot_W_M) / np.abs(dot_E_out))
     
    return
                                              
if __name__ == "__main__":
        
    test_contraction_relaxation(plot_control = False,
        make_video = False, active_clip = False, plot = True)
        
    print('Finished')






