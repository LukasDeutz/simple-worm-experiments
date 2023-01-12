'''
Created on 28 Jul 2022

@author: lukas
'''
# Build-in imports
from os.path import isfile, join
import pickle

# Third-party imports
import numpy as np
from fenics import Function, Expression

# Local imports
from simple_worm.rod.cosserat_rod_2_test import CosseratRod_2
from simple_worm.controls import ControlsFenics, ControlSequenceFenics
from simple_worm_experiments.util import default_solver, get_solver, dimensionless_MP, save_output
from simple_worm.util import v2f

from mp_progress_logger import FWException

data_path = "../../data/roll_maneuver/"
fig_path = "../../fig/experiments/roll_maneuver/"


class RollManeuverExperiment():
    '''
    Implements control sequences for different roll maneuver experiments
    '''
                                        
    @staticmethod
    def repetitious_roll(worm, parameter):
        '''
        Returns repetitious roll maneuver control sequence 

        :param worm (CosseratRod): worm object
        :param parameter (dict): parameter dictionary
        
        '''        
        # Read in parameters        
        T = parameter['T']
                
        smo, mts = parameter['smo'], parameter['mts']        
        
        A_dv, f_dv = parameter['A_dv'], parameter['f_dv']
        A_lr, f_lr = parameter['A_lr'], parameter['f_lr']
        
        w_dv = 2*np.pi*f_dv        
        w_lr = 2*np.pi*f_lr
        
        phi = parameter['phi']
                                                    
        if mts: #Muscles switch on and off on finite timescale            
            tau_on = parameter['tau_on']
            Dt_on = parameter['Dt_on']
            m_on = Expression('1 / ( 1 + exp(- ( t - Dt) / tau))', degree = 1, t = 0, Dt = Dt_on, tau = tau_on)
        else:
            m_on = Expression('1', degree = 1)
        
        if smo: # Smooth muscle onset        
            Ds = parameter['Ds']
            s0 = parameter['s0']            
            # sigmoid head
            sh = Expression('1 / ( 1 + exp(- (x[0] - s0) / Ds ) )', degree = 1, Ds=Ds, s0 = s0) 
            # sigmoid tale
            st = Expression('1 / ( 1 + exp(  (x[0] - 1 + s0) / Ds) )', degree = 1, Ds=Ds, s0 = s0)                                 
        else:
            st = Expression('1', degree = 1)
            sh = Expression('1', degree = 1)
               
        Ds_h = parameter['Ds_h']
        s0_h = parameter['s0_h']

        st_lr = Expression('1 / ( 1 + exp(  (x[0] - 1 + s0) / Ds) )', degree = 1, Ds=Ds_h, s0 = s0_h) 
        st_dv = Expression('1 / ( 1 + exp(  (x[0] - 1 + s0) / Ds) )', degree = 1, Ds=Ds_h, s0 = s0_h) 
                           
                           
        Omega_expr = Expression(("m_on*sh*st_dv*A1*sin(w1*t)", 
                                 "m_on*sh*st_lr*A2*sin(w2*t + phi)",
                                 "0"), 
                                 degree=1,
                                 t = 0,
                                 A1 = A_dv,
                                 w1 = w_dv,
                                 A2 = A_lr,
                                 w2 = w_lr,
                                 phi = phi,                                                                  
                                 st = st,
                                 sh = sh,
                                 st_lr = st_lr,
                                 st_dv = st_dv,
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
                    
    # @staticmethod
    # def undulation_roll_undulation(parameter, worm):
    #
    #     T = parameter['T']
    #     dt = parameter['dt']
    #
    #     t_arr = np.arange(dt, T+0.1*dt, dt)
    #
    #     smo = parameter['smo']
    #
    #     A0, A1 = parameter['A0'], parameter['A1']
    #     lam0, lam1  = parameter['lam0'], parameter['lam1']
    #     f0, f1 = parameter['f0'], parameter['f1']
    #
    #     Delta_t = parameter['Delta_t']
    #     t0 = parameter['t0']
    #
    #     k0 = (2 * np.pi) / lam0
    #     w0 = 2*np.pi*f0
    #     c0 = lam0*f0
    #
    #     k1 = (2 * np.pi) / lam1
    #     w1 = 2*np.pi*f1
    #     c1 = lam1*f1
    #
    #     C_arr = []
    #
    #     # If smo, then muscle forces are zero at the tip of the head and tale
    #     # Zero muscle forces equate to a zero curvature ampltiude 
    #     # Thus, we scale the constant curvature amplitude with a sigmoid at the tale and the head 
    #     if smo:            
    #         a  = parameter['a_smo']
    #         du = parameter['du_smo']
    #
    #         sig_t = 1.0/(1.0 + np.exp(-a*(self.s_arr - du)))
    #         sig_h = 1.0/(1.0 + np.exp(a*(self.s_arr - 1.0 + du))) 
    #
    #     for t in t_arr:
    #
    #
    #         # For t < 0, the worm does a standard forward undulation
    #         # and we approximate the curvature k as a sin wave
    #         if t < t0:
    #
    #             k_1_arr = A0*np.sin(k0*self.s_arr - w0 * t)
    #             k_2_arr = np.zeros(self.N)
    #             k_3_arr = np.zeros(self.N)
    #
    #             if smo:                                        
    #                 k_1_arr = sig_t * sig_h * k_1_arr
    #
    #         elif t >= t0 and t <= t0 + Delta_t:         
    #
    #             # At t=t0, the worm starts the roll maneuver.     
    #             # The roll control starts at the head and it replaces 
    #             # the undulation coontrol while it travels along the body 
    #             # to the tale. 
    #             s1 = c1*(t - t0)
    #
    #             idx1 = self.s_arr <= s1
    #             idx0 = self.s_arr >  s1
    #
    #             s_arr_1 = self.s_arr[idx1]
    #             s_arr_0 = self.s_arr[idx0]
    #
    #             #TODO: Roll maneuver goes here
    #             # k1 = alpha
    #             # k2 = beta
    #             # k3 = gamma
    #             N1 = len(s_arr_1)
    #
    #             k1_1_arr = 1*np.ones(N1)                                                
    #             k2_1_arr = np.zeros(N1)
    #             k3_1_arr = np.zeros(N1)
    #
    #             k1_0_arr = A0 * np.sin(k0*s_arr_0 - w0 * t)                    
    #             k2_0_arr = np.zeros(len(s_arr_0))
    #             k3_0_arr = np.zeros(len(s_arr_0))
    #
    #             if smo:                                        
    #                 k1_0_arr = sig_t[idx0] * sig_h[idx0] * k1_0_arr
    #                 k1_1_arr = sig_t[idx1] * sig_h[idx1] * k1_1_arr
    #
    #                 k2_1_arr = sig_t[idx1] * sig_h[idx1] * k2_1_arr
    #                 k3_1_arr = sig_t[idx1] * sig_h[idx1] * k3_1_arr
    #
    #             k_1_arr = np.concatenate((k1_1_arr, k1_0_arr))
    #             k_2_arr = np.concatenate((k2_1_arr, k2_0_arr))
    #             k_3_arr = np.concatenate((k3_1_arr, k3_0_arr))
    #
    #         elif t > t0+Delta_t:                
    #             # At t=t0+Delta_t, the worm has finished the roll maneuver 
    #             # and it changes its control back two a planar undulation.
    #             # The undulation control starts at the head and it replaces 
    #             # the roll maneuver control while it travels along the body 
    #             # to the tale.             
    #             s0 = c0*(t - t0 - Delta_t)
    #
    #             # If s0>1.0, then the worm has finished its turn maneuver and 
    #             # the curvature k is a sin wave  
    #             if s0 <= 1.0:
    #
    #                 s1 = c1*(t - t0)
    #
    #                 # If s1 <= 1.0, then the curvature function is composed of three 
    #                 # sine waves. The modified wave is sandwiched by two waves 
    #                 # at the head and tale which have the same kinematic parameters                    
    #                 if s1 <= 1.0:
    #
    #                     idx0_h = self.s_arr <= s0
    #                     idx1 = (self.s_arr > s0) & (self.s_arr <= s1)                
    #                     idx0_t = self.s_arr > s1
    #
    #                     s_arr_0_h = self.s_arr[idx0_h]
    #                     s_arr_1 = self.s_arr[idx1]
    #                     s_arr_0_t = self.s_arr[idx0_t]
    #
    #                     k_1_0_h_arr = A0 * np.sin(k0*s_arr_0_h - w0 * t)
    #                     k_2_0_h_arr = np.zeros(len(s_arr_0_h))
    #                     k_3_0_h_arr = np.zeros(len(s_arr_0_h))
    #
    #                     #TODO: Roll maneuver goes here
    #                     # k1 = alpha
    #                     # k2 = beta
    #                     # k3 = gamma
    #                     N1 = len(s_arr_1)
    #
    #                     k1_1_arr = 1*np.ones(N1)
    #                     k2_1_arr = np.zeros(N1)
    #                     k3_1_arr = np.zeros(N1)
    #
    #                     k_1_0_t_arr =  A0 * np.sin(k0*s_arr_0_t - w0 * t)
    #                     k_2_0_t_arr = np.zeros(len(s_arr_0_t))
    #                     k_3_0_t_arr = np.zeros(len(s_arr_0_t))
    #
    #                     if smo:
    #                         k_1_0_h_arr = sig_t[idx0_h] * sig_h[idx0_h] * k_1_0_h_arr
    #                         k1_1_arr = sig_t[idx1] * sig_h[idx1] * k1_1_arr
    #                         k_1_0_t_arr = sig_t[idx0_t] * sig_h[idx0_t] * k_1_0_t_arr
    #
    #                         k2_1_arr = sig_t[idx1] * sig_h[idx1] * k2_1_arr
    #                         k3_1_arr = sig_t[idx1] * sig_h[idx1] * k3_1_arr
    #
    #
    #                     k_1_arr = np.concatenate((k_1_0_h_arr, k1_1_arr, k_1_0_t_arr))
    #                     k_2_arr = np.concatenate((k_2_0_h_arr, k2_1_arr, k_2_0_t_arr))
    #                     k_3_arr = np.concatenate((k_3_0_h_arr, k3_1_arr, k_3_0_t_arr))
    #
    #                 # If s >= 1.0, then the modified curvature wave has reached 
    #                 # the tale, i.e. the curvature function is only composed of 
    #                 # two sine waves, one with the initial kinematic parameters
    #                 # which goes from head to s0 and the modified 
    #                 else:
    #
    #                     idx0 = self.s_arr <= s0
    #                     idx1 = self.s_arr > s0                 
    #
    #                     idx0 = self.s_arr <= s0
    #                     idx1 = self.s_arr > s0  
    #
    #                     s_arr_0 = self.s_arr[idx0]
    #                     s_arr_1 = self.s_arr[idx1]
    #
    #                     k1_0_arr = A0 * np.sin(k0*s_arr_0 - w0 * t)
    #                     k2_0_arr = np.zeros(len(s_arr_0))
    #                     k3_0_arr = np.zeros(len(s_arr_0))
    #
    #                     #TODO: Roll maneuver goes here
    #                     # k1 = alpha
    #                     # k2 = beta
    #                     # k3 = gamma  
    #
    #                     N1 = len(s_arr_1)                   
    #                     k1_1_arr = 1*np.ones(N1)
    #                     k2_1_arr = np.zeros(N1)
    #                     k3_1_arr = np.zeros(N1)
    #
    #                     if smo:
    #                         k1_0_arr = sig_t[idx0] * sig_h[idx0] * k1_0_arr
    #                         k1_1_arr = sig_t[idx1] * sig_h[idx1] * k1_1_arr
    #
    #                         k2_1_arr = sig_t[idx1] * sig_h[idx1] * k2_1_arr
    #                         k3_1_arr = sig_t[idx1] * sig_h[idx1] * k3_1_arr
    #
    #                     k_1_arr = np.concatenate((k1_0_arr, k1_1_arr))
    #                     k_2_arr = np.concatenate((k2_0_arr, k2_1_arr))
    #                     k_3_arr = np.concatenate((k3_0_arr, k3_1_arr))
    #
    #             # Roll maneuver is finished the control along the entire body 
    #             # is simply given by an undulation
    #             else:
    #
    #                 k_1_arr = A0*np.sin(k0*self.s_arr - w0 * t)
    #                 k_2_arr = np.zeros(self.N)
    #                 k_3_arr = np.zeros(self.N)
    #
    #                 if smo:                    
    #
    #                     k_1_arr = sig_t * sig_h * k_1_arr
    #
    #
    #         # TODO: Add k2=beta and k3=gamma if neccessary                                                                                            
    #         k_arr = np.vstack((k_1_arr, k_2_arr, k_3_arr))
    #
    #         k_func = Function(self.worm.function_spaces['Omega'])            
    #         k_func = v2f(k_arr, k_func, W = self.worm.function_spaces['Omega'])
    #
    #         C = ControlsFenics(k_func, self.sig_func)
    #
    #         C_arr.append(C)
    #
    #     CS = ControlSequenceFenics(C_arr)
    #     CS.rod  = self.worm
    #
    #     return CS


   
    

        
        



