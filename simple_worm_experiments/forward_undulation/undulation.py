# Third party imports
from fenics import Expression, Function
import numpy as np

# Local imports
from simple_worm.controls import ControlsFenics, ControlSequenceFenics
from simple_worm_experiments.experiment import Experiment 
        
#===============================================================================
# Simulate undulation

class UndulationExperiment(Experiment):
    '''
    Implements control sequences to model simple 2d undulation gait
    '''
    
    @staticmethod                                            
    def sinusoidal_traveling_wave_control_sequence(worm, parameter):
        '''
        Returns forward undualtion control sequence 

        :param worm (CosseratRod): worm object
        :param parameter (dict): parameter dictionary
        '''
                
        # Read in parameters
        
        T = parameter['T']
        # Kinematic parameter
        lam, f = parameter['lam'], parameter['f']                
        w, q = 2*np.pi*f, 2*np.pi/lam
        
        if parameter['A'] is not None:
            A = parameter['A']            
        else:
            c = parameter['c']
            A = c*q
                                    
        # Muscles switch on and off on a finite time scale                
        if parameter['fmts']:        
            tau_on, Dt_on  = parameter['tau_on'], parameter['Dt_on']
            sm_on = UndulationExperiment.sig_m_on_expr(tau_on, Dt_on)
        else:
            sm_on = Expression('1', degree = 1)
            
        # Gradual muscle activation onset at head and tale
        if parameter['gmo']:                            
            Ds_h, s0_h = parameter['Ds_h'], parameter['s0_h']
            Ds_t, s0_t = parameter['Ds_t'], parameter['s0_t']
            sh = UndulationExperiment.sig_head_expr(Ds_h, s0_h)
            st = UndulationExperiment.sig_tale_expr(Ds_t, s0_t)
        else: 
            sh = Expression('1', degree = 1)
            st = Expression('1', degree = 1)
                                                                
        Omega_expr = Expression(("sm_on*sh*st*A*sin(q*x[0] - w*t)", "0", "0"), 
            degree=1, t = 0, A = A, q = q, w = w, st = st, sh = sh, sm_on = sm_on)   
                  
        sigma_expr = Expression(('0', '0', '0'), degree = 1)    
        sigma = Function(worm.function_spaces['sigma'])
        sigma.assign(sigma_expr)
                                                  
        CS = []
            
        for t in np.linspace(0, T, int(T/worm.dt)):
            
            sm_on.t = t
            Omega_expr.t = t                                                                  
            Omega = Function(worm.function_spaces['Omega'])        
            Omega.assign(Omega_expr)
                                    
            C = ControlsFenics(Omega, sigma)
            CS.append(C)
                        
        CS = ControlSequenceFenics(CS)
        CS.rod = worm
            
        return CS  
