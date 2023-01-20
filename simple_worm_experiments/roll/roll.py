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
from simple_worm.controls import ControlsFenics, ControlSequenceFenics, ControlSequenceNumpy

from mp_progress_logger import FWException
from abc import abstractmethod

data_path = "../../data/roll_maneuver/"
fig_path = "../../fig/experiments/roll_maneuver/"


class RollManeuverExperiment():
    '''
    Implements control sequences for different roll maneuver experiments
    '''
       
    @staticmethod   
    def sig_m_on(t0, tau):
        
        return lambda t: 1.0 / (1 + np.exp(-( t - t0 ) / tau))
    
    @staticmethod   
    def sig_m_off(t0, tau):

        return lambda t: 1.0 / (1 + np.exp( ( t - t0) / tau))

                                        
    @staticmethod
    def continuous_roll(worm, parameter):
        '''
        Returns continuous roll maneuver control sequence 

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

    @staticmethod
    def discrete_roll_curved_worm(worm, parameter):

        '''
        Returns discrete roll maneuver control sequence 

        :param worm (CosseratRod): worm object
        :param parameter (dict): parameter dictionary
        
        '''        
                
        # Read in parameters        
        T, T_init = parameter['T'], parameter['T_init']        
        N, dt = parameter['N'], parameter['dt']

        t_arr = np.arange(dt, T + 0.1*dt, dt)
        n = len(t_arr)
        s_arr = np.linspace(0, 1, N)
        
        # gradual muscle onset at the tip of the head
        # and the tale
        smo = parameter['smo'] 

        # Assume worm is in a static planar configuration 
        # with sinusoidal curvature 
        A_static = parameter['A_static']
        lam_static = parameter['lam_static']
        q_static = 2 * np.pi / lam_static
        
        k_arr = np.zeros((3, N))
        k_arr[0, :] = A_static*np.sin(q_static*s_arr)
 
        # Worm rolls by rotating its head and neck region
        Dt_roll = parameter['Dt_roll']
                
        A_dv, f_dv = parameter['A_dv'], parameter['f_dv']
        A_lr, f_lr = parameter['A_lr'], parameter['f_lr']
        
        w_dv = 2*np.pi*f_dv        
        w_lr = 2*np.pi*f_lr
        
        phi = parameter['phi']

        # For the roll, muscles are only activated in the head and neck
        Ds_h, s0_h= parameter['Ds_h'], parameter['s0_h']
        
        st_lr = 1.0 / (1.0 + np.exp((s_arr - 1 + s0_h) / Ds_h ) )
        st_dv = 1.0 / (1.0 + np.exp((s_arr - 1 + s0_h) / Ds_h ) )
                                                                                                                                    
        if smo: # No muscles at the tip of the head and tale             
            Ds = parameter['Ds']
            s0 = parameter['s0']            
            # sigmoid head
            sh = 1.0 / (1.0 + np.exp(-(s_arr - s0) / Ds))
            # sigmoid tale
            st = 1.0 / (1.0 + np.exp( (s_arr - 1 + s0) / Ds))
            
            k_arr[0, :] = sh * st * k_arr[0, :]
        
        #Muscles switch on and off on finite timescale                                                            
        tau_on = parameter['tau_on']
        tau_off = parameter['tau_off']            
        Dt_on = parameter['Dt_on']
        Dt_off = parameter['Dt_off']

        m_static_on = lambda t: 1.0 / (1 + np.exp(-( t - Dt_on ) / tau_on))
        m_roll_on  = lambda t: 1.0 / (1 + np.exp(-( t - T_init + Dt_on ) / tau_on))
        m_roll_off = lambda t: 1.0 / (1 + np.exp( ( t - T_init - Dt_roll - Dt_off) / tau_off))
                                                                                                                                                      
        # Add time dimension
        k_arr = k_arr[None, :, :]        
        k_arr = np.vstack(n*[k_arr])
                                
        for i, t in enumerate(t_arr):
            
            k_arr[i, 0] = k_arr[i, 0] * m_static_on(t)
                        
            k1_r = st_dv * A_dv * np.sin(w_dv*t)
            k2_r = st_lr * A_lr * np.sin(w_lr*t + phi)
                    
            k1_r = m_roll_on(t) * m_roll_off(t) * m_roll_off(t) * k1_r
            k2_r = m_roll_on(t) * m_roll_off(t) * m_roll_off(t) * k2_r                                                        
            
            if smo:
                k1_r = sh * st * k1_r
                k2_r = sh * st * k2_r                    
                                                                                
            k_arr[i, 0, :] += k1_r
            k_arr[i, 1, :] += k2_r
                                                                
        sig_arr = np.zeros((n, 3, N))

        CSN = ControlSequenceNumpy(Omega=k_arr, sigma=sig_arr)
        CS = CSN.to_fenics(worm)
        CS.rod = worm    
        
        return CS

    @staticmethod
    def discrete_roll_straight_worm(worm, parameter):
        '''
        Returns discrete roll maneuver control sequence
        
        Control is inspired by calcium imaging videos
        
        :param worm (CosseratRod): worm object
        :param parameter (dict): parameter dictionary
        '''
                
        # Read in parameters        
        T, Dt1, Dt2 = parameter['T'], parameter['Dt1'], parameter['Dt2']     
        N, dt = parameter['N'], parameter['dt']
                        
        t_arr = np.arange(dt, T + 0.1*dt, dt)
        n = len(t_arr)
        s_arr = np.linspace(0, 1, N)
        
        # gradual muscle onset at the tip of the head
        # and the tale
        smo = parameter['smo'] 
         
        # Worm rolls by rotating its head and neck region
        Dt_roll = parameter['Dt_roll']
                
        A_dv, f_dv = parameter['A_dv'], parameter['f_dv']
        A_lr, f_lr = parameter['A_lr'], parameter['f_lr']
        
        w_dv = 2*np.pi*f_dv        
        w_lr = 2*np.pi*f_lr
        
        phi = parameter['phi']

        # For the roll, muscles are only activated in the head and neck
        Ds_h, s0_h= parameter['Ds_h'], parameter['s0_h']
        
        st_lr = 1.0 / (1.0 + np.exp((s_arr - 1 + s0_h) / Ds_h ) )
        st_dv = 1.0 / (1.0 + np.exp((s_arr - 1 + s0_h) / Ds_h ) )
                                                                                                                                    
        if smo: # No muscles at the tip of the head and tale             
            Ds = parameter['Ds']
            s0 = parameter['s0']            
            # sigmoid head
            sh = 1.0 / (1.0 + np.exp(-(s_arr - s0) / Ds))
            # sigmoid tale
            st = 1.0 / (1.0 + np.exp( (s_arr - 1 + s0) / Ds))
                    
        #Muscles switch on and off on finite timescale                                                            
        tau_on = parameter['tau_on']
        tau_off = parameter['tau_off']            
        Dt_on = parameter['Dt_on']
        Dt_off = parameter['Dt_off']

        m_static_on = lambda t: 1.0 / (1 + np.exp(-( t - Dt_on ) / tau_on))
        
        
        
        
        m_roll_off = lambda t: 1.0 / (1 + np.exp( ( t - T_init - Dt_roll - Dt_off) / tau_off))
                                                                                                                                                      
        # Add time dimension
        k_arr = k_arr[None, :, :]        
        k_arr = np.vstack(n*[k_arr])
                                
        for i, t in enumerate(t_arr):
            
            k_arr[i, 0] = k_arr[i, 0] * m_static_on(t)
                        
            k1_r = st_dv * A_dv * np.sin(w_dv*t)
            k2_r = st_lr * A_lr * np.sin(w_lr*t + phi)
                    
            k1_r = m_roll_on(t) * m_roll_off(t) * m_roll_off(t) * k1_r
            k2_r = m_roll_on(t) * m_roll_off(t) * m_roll_off(t) * k2_r                                                        
            
            if smo:
                k1_r = sh * st * k1_r
                k2_r = sh * st * k2_r                    
                                                                                
            k_arr[i, 0, :] += k1_r
            k_arr[i, 1, :] += k2_r
                                                                
        sig_arr = np.zeros((n, 3, N))

        CSN = ControlSequenceNumpy(Omega=k_arr, sigma=sig_arr)
        CS = CSN.to_fenics(worm)
        CS.rod = worm    
        
        return CS        
        
            
            
            
            
            
            
            
        
        
        


                    
   


   
    

        
        



