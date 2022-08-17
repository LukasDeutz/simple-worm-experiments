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
                            
        A0 = parameter['A0'], A1 = parameter['A0']
        lam0 = parameter['lam0'], lam1 = parameter['lam1']
        f0 = parameter['f0'], f1 = parameter['f1']
        
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
            
            st = 1.0/(1.0 + np.exp(-a*(self.s_arr - du)))
            sh = 1.0/(1.0 + np.exp(-a*(self.s_arr - 1.0 + du))) 
                            
        for t in t_arr:

            if t < t0:

                k_arr = A0*np.sin(k0*self.s_arr - w0 * t)

                if smo:                    
                    
                    k_arr = st * sh * k_arr
            
            elif t >= t0 and t <= t0 + Delta_t:         
                
                s1 = c1*(t - t0)
                                                
                s_arr_1 = self.s_arr[self.s_arr <= s1]
                s_arr_0 = self.s_arr[self.s_arr >  s1]

                k1_arr = A1 * np.sin(k1*s_arr_1 - w1 * t)
                k0_arr = A0 * np.sin(k0*s_arr_0 - w0 * t)                    

                if smo:
                    k1_arr = st * sh * k1_arr
                    k0_arr = st * sh * k0_arr
                
                k_arr = np.concatenate((k1_arr, k0_arr))

            else:                
                
                s0 = c0*(t - t0 - Delta_t)

                s_arr_0 = self.s_arr[self.s_arr <= s0]
                s_arr_1 = self.s_arr[self.s_arr > s0]

                k0_arr = A0 * np.sin(k0*s_arr_0 - w0 * t)
                k1_arr = A1 * np.sin(k1*s_arr_1 - w1 * t)

                if smo:
                    k1_arr = st * sh * k1_arr
                    k0_arr = st * sh * k0_arr

                k_arr = np.concatenate((k0_arr, k1_arr))
                                                
            k_arr = np.vstack((k_arr, np.zeros(self.N), np.zeros(self.N)))
            
            k_func = Function(self.worm.function_spaces['Omega'])            
            k_func = v2f(k_arr, k_func, W = self.worm.function_spaces['Omega'])
                        
            C = ControlsFenics(k_func, self.sig_func)
                                            
            C_arr.append(C)
        
        CS = ControlSequenceFenics(C_arr)
        CS.rod  = self.worm
        
        return CS
    
    def simulate_planar_turn(self, 
                             parameter, 
                             ):            


        T = parameter['T']
                                                                                                                                    
        MP = dimensionless_MP(parameter)
        CS = self.undulation_control_sequence(parameter) 
                            
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
        FS, CS, MP = PTE.simulate_undulation(parameter)
    
    else:    
        try:        
            FS, CS, MP = PTE.simulate_undulation(parameter)
        except Exception as e:
            traceback.print_exception(e)                
            with open(data_path + '/errors/' + 'error_report_' + _hash + '.json', 'w') as f:        
                json.dump(parameter, f, indent=4)    
            return
        
    save_output(FS, MP, CS, parameter, save)

    return
    
    
    



