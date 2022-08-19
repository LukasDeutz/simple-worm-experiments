'''
Created on 25 May 2022

@author: lukas
'''
# Build-in imports
import copy
import itertools as it
from os.path import isfile
from multiprocessing import Pool
import json
import traceback
import pickle

# Third-party imports
import numpy as np
from fenics import Function, Expression

# Local imports
from simple_worm.rod.cosserat_rod_2_test import CosseratRod_2
from simple_worm.controls import ControlsFenics, ControlSequenceFenics
from simple_worm_experiments.util import default_solver, get_solver, dimensionless_MP, save_output
from simple_worm.util import v2f

data_path = "../../data/planar_turn/"


fig_path = "../../fig/experiments/planar_turn/"

class PlanarTurnExperiment():

    def __init__(self, N, dt, solver = None):
    
    
        self.N = N
        self.dt = dt
    
        if solver is None:
            solver = default_solver()
            
        self.solver = solver

        self.worm = CosseratRod_2(self.N, dt, self.solver)                                                        
        

        self.s_arr = np.linspace(0, 1, self.N)


        self.sig_arr = np.zeros((3, self.N))
        self.sig_func = Function(self.worm.function_spaces['sigma'])
        self.sig_func.assign(Expression(('0', '0', '0'), degree = 1))
            
        return
                                                            
    def planar_turn_CS(self, parameter):


        T = parameter['T']

        t_arr = np.arange(self.dt, T+0.1*self.dt, self.dt)

        smo = parameter['smo']
                            
        A0, A1 = parameter['A0'], parameter['A1']
        lam0, lam1  = parameter['lam0'], parameter['lam1']
        f0, f1 = parameter['f0'], parameter['f1']
        
        Delta_t = parameter['Delta_t']
        t0 = parameter['t0']
                                
        k0 = (2 * np.pi) / lam0
        w0 = 2*np.pi*f0
        c0 = lam0*f0

        k1 = (2 * np.pi) / lam1
        w1 = 2*np.pi*f1
        c1 = lam1*f1
        
        C_arr = []
        
        # If smo, then muscle forces are zero at the tip of the head and tale
        # Zero muscle forces equate to a zero curvature ampltiude 
        # Thus, we scale the constant curvature amplitude with a sigmoid at the tale and the head 
        if smo:            
            a  = parameter['a_smo']
            du = parameter['du_smo']
            
            sig_t = 1.0/(1.0 + np.exp(-a*(self.s_arr - du)))
            sig_h = 1.0/(1.0 + np.exp(a*(self.s_arr - 1.0 + du))) 
                            
        for t in t_arr:


            # For t < 0, the worm does a standard forward undulation
            # and we approximate the curvature k as a sin wave
            if t < t0:

                k_arr = A0*np.sin(k0*self.s_arr - w0 * t)

                if smo:                    
                    
                    k_arr = sig_t * sig_h * k_arr
            
            elif t >= t0 and t <= t0 + Delta_t:         
                
                # At t=t0, the worm starts the turn maneuver.     
                # The worm turns by changing the amplitude, frequency
                # wavelength of curvature the wave function in a step-wise 
                # manner. The modified curvature wave starts at head 
                # and it replaces the previous wave while it travels
                # along to body to the tale. 
                s1 = c1*(t - t0)
                    
                idx1 = self.s_arr <= s1
                idx0 = self.s_arr >  s1
                                                
                s_arr_1 = self.s_arr[idx1]
                s_arr_0 = self.s_arr[idx0]

                k1_arr = A1 * np.sin(k1*s_arr_1 - w1 * t)
                k0_arr = A0 * np.sin(k0*s_arr_0 - w0 * t)                    

                if smo:                                        
                    k0_arr = sig_t[idx0] * sig_h[idx0] * k0_arr
                    k1_arr = sig_t[idx1] * sig_h[idx1] * k1_arr
                
                k_arr = np.concatenate((k1_arr, k0_arr))

            elif t > t0+Delta_t:                
                # At t=t0+Delta_t, the worm has finished the turn maneuver 
                # and it changes the amplitude,frequency and wavelength of 
                # curvature back to their initial values. Again, the changed
                # curvature wave starts at head it replaces the modified wave 
                # while it travels along to body to the tale.             
                s0 = c0*(t - t0 - Delta_t)
                
                # If s0>1.0, then the worm has finished its turn maneuver and 
                # the curvature k is a sin wave  
                if s0 <= 1.0:
                                                                                
                    s1 = c1*(t - t0)
                                        
                    # If s1 <= 1.0, then the curvature function is composed of three 
                    # sine waves. The modified wave is sandwiched by two waves 
                    # at the head and tale which have the same kinematic parameters                    
                    if s1 <= 1.0:
                                                        
                        idx0_h = self.s_arr <= s0
                        idx1 = (self.s_arr > s0) & (self.s_arr <= s1)                
                        idx0_t = self.s_arr > s1
                                    
                        s_arr_0_h = self.s_arr[idx0_h]
                        s_arr_1 = self.s_arr[idx1]
                        s_arr_0_t = self.s_arr[idx0_t]
        
                        k0_h_arr = A0 * np.sin(k0*s_arr_0_h - w0 * t)
                        k1_arr = A1 * np.sin(k1*s_arr_1 - w1 * t)
                        k0_t_arr =  A0 * np.sin(k0*s_arr_0_t - w0 * t)
    
                        if smo:
                            k0_h_arr = sig_t[idx0_h] * sig_h[idx0_h] * k0_h_arr
                            k1_arr = sig_t[idx1] * sig_h[idx1] * k1_arr
                            k0_t_arr = sig_t[idx0_t] * sig_h[idx0_t] * k0_t_arr
    
                        k_arr = np.concatenate((k0_h_arr, k1_arr, k0_t_arr))
                        
                    # If s >= 1.0, then the modified curvature wave has reached 
                    # the tale, i.e. the curvature function is only composed of 
                    # two sine waves, one with the initial kinematic parameters
                    # which goes from head to s0 and the modified 
                    else:
                        
                        idx0 = self.s_arr <= s0
                        idx1 = self.s_arr > s0                 

                        idx0 = self.s_arr <= s0
                        idx1 = self.s_arr > s0  
                                    
                        s_arr_0 = self.s_arr[idx0]
                        s_arr_1 = self.s_arr[idx1]
        
                        k0_arr = A0 * np.sin(k0*s_arr_0 - w0 * t)
                        k1_arr = A1 * np.sin(k1*s_arr_1 - w1 * t)
    
                        if smo:
                            k0_arr = sig_t[idx0] * sig_h[idx0] * k0_arr
                            k1_arr = sig_t[idx1] * sig_h[idx1] * k1_arr
    
                        k_arr = np.concatenate((k0_arr, k1_arr))
                
                else:
                    
                    k_arr = A0*np.sin(k0*self.s_arr - w0 * t)
    
                    if smo:                    
                        
                        k_arr = sig_t * sig_h * k_arr
                                                                                            
            k_arr = np.vstack((k_arr, np.zeros(self.N), np.zeros(self.N)))
            
            k_func = Function(self.worm.function_spaces['Omega'])            
            k_func = v2f(k_arr, k_func, W = self.worm.function_spaces['Omega'])
                        
            C = ControlsFenics(k_func, self.sig_func)
                                            
            C_arr.append(C)
        
        CS = ControlSequenceFenics(C_arr)
        CS.rod  = self.worm
        
        return CS
    
    def simulate_planar_turn(self, parameter):            

        T = parameter['T']
                                                                                                                                    
        MP = dimensionless_MP(parameter)
        CS = self.planar_turn_CS(parameter)
                            
        FS = self.worm.solve(T, CS=CS, MP = MP) 
                                  
        return FS, CS, MP

def wrap_simulate_planar_turn(parameter, data_path, _hash, overwrite = False, save = 'min', _try = False):

    fn = _hash 

    if not overwrite:    
        if isfile(data_path + 'simulations/' + fn + '.dat'):
            print('File already exists')
            return
        
    PTE = PlanarTurnExperiment(parameter['N'], parameter['dt'], solver = get_solver(parameter))
                
    if not _try:
        FS, CS, MP = PTE.simulate_planar_turn(parameter)
    
    else:    
        try:        
            FS, CS, MP = PTE.simulate_planar_turn(parameter)
        except Exception as e:
            traceback.print_exception(e)                
            with open(data_path + '/errors/' + 'error_report_' + _hash + '.json', 'w') as f:        
                json.dump(parameter, f, indent=4)    
            return
        
    save_output(FS, MP, CS, parameter, save)

    return
    
    
    



