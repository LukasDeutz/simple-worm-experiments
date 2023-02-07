'''
Created on 18 Jan 2023

@author: lukas
'''
# Third-party imports
import numpy as np
from fenics import Expression, Function

# Local imports
from simple_worm_experiments.experiment import Experiment 
from simple_worm.controls import ControlsFenics, ControlSequenceFenics#, ControlSequenceNumpy

class CoilingExperiment(Experiment):

    @staticmethod
    def continuous_coiling(worm, parameter):
        '''
        Returns continuous coil maneuver control sequence 

        :param worm (CosseratRod): worm object
        :param parameter (dict): parameter dictionary
        
        '''        
        # Read in parameters        
        T = parameter['T']
                
        mts = parameter['mts']        
        
        A_dv, f_dv, lam_dv = parameter['A_dv'], parameter['f_dv'], parameter['lam_dv'] 
        A_lr, f_lr, lam_lr = parameter['A_lr'], parameter['f_lr'], parameter['lam_lr']
        
        w_dv = 2*np.pi*f_dv        
        w_lr = 2*np.pi*f_lr
        q_dv = 2*np.pi / lam_dv
        q_lr = 2*np.pi / lam_lr
        
        phi = parameter['phi']
              
        # Muscles switch on and off
        # on a finit time scale
        if mts: 
            tau_on = parameter['tau_on']
            Dt_on = parameter['Dt_on']
            m_on = CoilingExperiment.sig_m_on_expr(Dt_on, tau_on)
        else:
            m_on = Expression('1', degree = 1)
                                                       
        Ds_dv_h = parameter['Ds_dv_h']
        s0_dv_h = parameter['s0_dv_h']
        Ds_dv_t = parameter['Ds_dv_t']
        s0_dv_t = parameter['s0_dv_t']

        Ds_lr_h = parameter['Ds_lr_h']
        s0_lr_h = parameter['s0_lr_h']
        Ds_lr_t = parameter['Ds_lr_t']
        s0_lr_t = parameter['s0_lr_t']

        if Ds_dv_h is not None:
            sh_dv = CoilingExperiment.sig_head_expr(Ds_dv_h, s0_dv_h)
        else: 
            sh_dv = Expression('1', degree = 1)        
        if Ds_dv_t is not None:            
            st_dv = CoilingExperiment.sig_tale_expr(Ds_dv_t, s0_dv_t)        
        else: 
            st_dv = Expression('1', degree = 1)

        if Ds_lr_h is not None:
            sh_lr = CoilingExperiment.sig_head_expr(Ds_lr_h, s0_lr_h)
        else: 
            sh_lr = Expression('1', degree = 1)        
        if Ds_lr_t is not None:            
            st_lr = CoilingExperiment.sig_tale_expr(Ds_lr_t, s0_lr_t)        
        else: 
            st_lr = Expression('1', degree = 1)
                                                              
        Omega_expr = Expression((
            "m_on*sh_dv*st_dv*A1*sin(q1*x[0] - w1*t)", 
            "m_on*sh_lr*st_lr*A2*sin(q2*x[0] - w2*t + phi)", 
            "0"), 
            degree=1, 
            t = 0, 
            A1 = A_dv,
            w1 = w_dv,
            q1 = q_dv,
            A2 = A_lr,
            w2 = w_lr,
            q2 = q_lr,
            phi = phi,                                                                  
            sh_dv = sh_dv,            
            sh_lr = sh_lr,
            st_dv = st_dv,            
            st_lr = st_lr,
            m_on = m_on)   
                                
        CS = []

        sigma_expr = Expression(('0', '0', '0'), degree = 1)    
        sigma = Function(worm.function_spaces['sigma'])
        sigma.assign(sigma_expr)
            
        for t in np.linspace(0, T, int(T/worm.dt)):
            
            m_on.t = t                                                                  
            Omega_expr.t = t
            Omega = Function(worm.function_spaces['Omega'])        
            Omega.assign(Omega_expr)
                                    
            C = ControlsFenics(Omega, sigma)
            CS.append(C)
                        
        CS = ControlSequenceFenics(CS)
        CS.rod = worm
            
        return CS  
