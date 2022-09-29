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
    
    def __init__(self, N, dt, solver = None, quiet = False):
        
        self.N = N
        self.dt = dt
        
        if solver is None:
            solver = default_solver()
            
        self.solver = solver
        self.worm = CosseratRod_2(self.N, dt, self.solver, quiet=quiet)                                                        
        
        self.s_arr = np.linspace(0, 1, self.N)

        self.sig_arr = np.zeros((3, self.N))
        self.sig_func = Function(self.worm.function_spaces['sigma'])
        self.sig_func.assign(Expression(('0', '0', '0'), degree = 1))


    def roll_maneuver_CS(self, parameter):

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

                k_1_arr = A0*np.sin(k0*self.s_arr - w0 * t)
                k_2_arr = np.zeros(self.N)
                k_3_arr = np.zeros(self.N)

                if smo:                                        
                    k_1_arr = sig_t * sig_h * k_1_arr
            
            elif t >= t0 and t <= t0 + Delta_t:         
                
                # At t=t0, the worm starts the roll maneuver.     
                # The roll control starts at the head and it replaces 
                # the undulation coontrol while it travels along the body 
                # to the tale. 
                s1 = c1*(t - t0)
                    
                idx1 = self.s_arr <= s1
                idx0 = self.s_arr >  s1
                                                
                s_arr_1 = self.s_arr[idx1]
                s_arr_0 = self.s_arr[idx0]

                #TODO: Roll maneuver goes here
                # k1 = alpha
                # k2 = beta
                # k3 = gamma
                N1 = len(s_arr_1)
                
                k1_1_arr = 1*np.ones(N1)                                                
                k2_1_arr = np.zeros(N1)
                k3_1_arr = np.zeros(N1)
                                
                k1_0_arr = A0 * np.sin(k0*s_arr_0 - w0 * t)                    
                k2_0_arr = np.zeros(len(s_arr_0))
                k3_0_arr = np.zeros(len(s_arr_0))

                if smo:                                        
                    k1_0_arr = sig_t[idx0] * sig_h[idx0] * k1_0_arr
                    k1_1_arr = sig_t[idx1] * sig_h[idx1] * k1_1_arr

                    k2_1_arr = sig_t[idx1] * sig_h[idx1] * k2_1_arr
                    k3_1_arr = sig_t[idx1] * sig_h[idx1] * k3_1_arr
                
                k_1_arr = np.concatenate((k1_1_arr, k1_0_arr))
                k_2_arr = np.concatenate((k2_1_arr, k2_0_arr))
                k_3_arr = np.concatenate((k3_1_arr, k3_0_arr))

            elif t > t0+Delta_t:                
                # At t=t0+Delta_t, the worm has finished the roll maneuver 
                # and it changes its control back two a planar undulation.
                # The undulation control starts at the head and it replaces 
                # the roll maneuver control while it travels along the body 
                # to the tale.             
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
        
                        k_1_0_h_arr = A0 * np.sin(k0*s_arr_0_h - w0 * t)
                        k_2_0_h_arr = np.zeros(len(s_arr_0_h))
                        k_3_0_h_arr = np.zeros(len(s_arr_0_h))
                                                
                        #TODO: Roll maneuver goes here
                        # k1 = alpha
                        # k2 = beta
                        # k3 = gamma
                        N1 = len(s_arr_1)
                        
                        k1_1_arr = 1*np.ones(N1)
                        k2_1_arr = np.zeros(N1)
                        k3_1_arr = np.zeros(N1)
                                                                                    
                        k_1_0_t_arr =  A0 * np.sin(k0*s_arr_0_t - w0 * t)
                        k_2_0_t_arr = np.zeros(len(s_arr_0_t))
                        k_3_0_t_arr = np.zeros(len(s_arr_0_t))

                        if smo:
                            k_1_0_h_arr = sig_t[idx0_h] * sig_h[idx0_h] * k_1_0_h_arr
                            k1_1_arr = sig_t[idx1] * sig_h[idx1] * k1_1_arr
                            k_1_0_t_arr = sig_t[idx0_t] * sig_h[idx0_t] * k_1_0_t_arr
    
                            k2_1_arr = sig_t[idx1] * sig_h[idx1] * k2_1_arr
                            k3_1_arr = sig_t[idx1] * sig_h[idx1] * k3_1_arr

    
                        k_1_arr = np.concatenate((k_1_0_h_arr, k1_1_arr, k_1_0_t_arr))
                        k_2_arr = np.concatenate((k_2_0_h_arr, k2_1_arr, k_2_0_t_arr))
                        k_3_arr = np.concatenate((k_3_0_h_arr, k3_1_arr, k_3_0_t_arr))
                                                
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
        
                        k1_0_arr = A0 * np.sin(k0*s_arr_0 - w0 * t)
                        k2_0_arr = np.zeros(len(s_arr_0))
                        k3_0_arr = np.zeros(len(s_arr_0))
                                                
                        #TODO: Roll maneuver goes here
                        # k1 = alpha
                        # k2 = beta
                        # k3 = gamma  
                        
                        N1 = len(s_arr_1)                   
                        k1_1_arr = 1*np.ones(N1)
                        k2_1_arr = np.zeros(N1)
                        k3_1_arr = np.zeros(N1)
                                                        
                        if smo:
                            k1_0_arr = sig_t[idx0] * sig_h[idx0] * k1_0_arr
                            k1_1_arr = sig_t[idx1] * sig_h[idx1] * k1_1_arr
    
                            k2_1_arr = sig_t[idx1] * sig_h[idx1] * k2_1_arr
                            k3_1_arr = sig_t[idx1] * sig_h[idx1] * k3_1_arr

                        k_1_arr = np.concatenate((k1_0_arr, k1_1_arr))
                        k_2_arr = np.concatenate((k2_0_arr, k2_1_arr))
                        k_3_arr = np.concatenate((k3_0_arr, k3_1_arr))
                
                # Roll maneuver is finished the control along the entire body 
                # is simply given by an undulation
                else:
                    
                    k_1_arr = A0*np.sin(k0*self.s_arr - w0 * t)
                    k_2_arr = np.zeros(self.N)
                    k_3_arr = np.zeros(self.N)
                        
                    if smo:                    
                        
                        k_1_arr = sig_t * sig_h * k_1_arr


            # TODO: Add k2=beta and k3=gamma if neccessary                                                                                            
            k_arr = np.vstack((k_1_arr, k_2_arr, k_3_arr))
            
            k_func = Function(self.worm.function_spaces['Omega'])            
            k_func = v2f(k_arr, k_func, W = self.worm.function_spaces['Omega'])
                        
            C = ControlsFenics(k_func, self.sig_func)
                                            
            C_arr.append(C)
                    
        CS = ControlSequenceFenics(C_arr)
        CS.rod  = self.worm
        
        return CS


    def simulate_roll_maneuver(self, parameter, pbar = None, logger = None):            

        T = parameter['T']
        N_report = parameter['N_report']
        dt_report = parameter['dt_report']
                                                                                                                                            
        MP = dimensionless_MP(parameter)
        CS = self.roll_maneuver_CS(parameter)
        
        #TODO: Do we need an initial configuration?
        F0 = None
        
        FS, e = self.worm.solve(T, MP, CS, F0, pbar, logger, N_report, dt_report)                                             
        
        CS = CS.to_numpy()
                                              
        return FS, CS, MP, e

def wrap_simulate_roll_maneuver(_input, 
                              pbar,
                              logger,
                              task_number,
                              output_dir, 
                              overwrite = False, 
                              save_keys = None,
                              ):
    
    parameter, _hash = _input[0], _input[1]
    
    filepath = join(output_dir, _hash + '.dat')
            
    if not overwrite:    
        if isfile(filepath):
            logger.info(f'Task {task_number}: File already exists')                                    
            output = pickle.load(open(join(output_dir, _hash + '.dat'), 'rb'))
            FS = output['FS']
                        
            exit_status = output['exit_status'] 
                        
            if exit_status:
                raise FWException(FS.pic, parameter['T'], parameter['dt'], FS.times[-1])
            
            result = {}
            result['pic'] = FS.pic
            
            return result
    
    N  = parameter['N']
    dt = parameter['dt']
        
    RME = RollManeuverExperiment(N, dt, solver = get_solver(parameter), quiet = True)
                    
    FS, CS, MP, e = RME.simulate_roll_maneuver(parameter, pbar, logger)
    
    if e is not None:
        exit_status = 1
    else:
        exit_status = 0
                
    # Regardless if simulation has finished or failed, simulation results
    # up to this point are saved to file         
    save_output(filepath, FS, CS, MP, parameter, exit_status, save_keys)                        
    logger.info(f'Task {task_number}: Saved file to {filepath}.')         
                
    # If the simulation has failed then we reraise the exception
    # which has been passed upstream        
    if e is not None:                
        raise FWException(FS.pic, 
                          parameter['T'], 
                          parameter['dt'], 
                          FS.times[-1]) from e
        
    # If simulation has finished succesfully then we return the relevant results 
    # for the logger
    result = {}    
    result['pic'] = FS.pic
    
    return result    
    

        
        



