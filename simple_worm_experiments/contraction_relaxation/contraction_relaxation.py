'''
Created on 2 Nov 2022

@author: lukas
'''

# Build-in imports
import copy
from os.path import isfile, join
import pickle

# Third party imports
from fenics import Expression, Function
import numpy as np

# Local imports
from simple_worm.rod.cosserat_rod_2_test import CosseratRod_2
from simple_worm.util import v2f
from simple_worm.controls.controls import ControlsFenics 
from simple_worm.controls.control_sequence import ControlSequenceFenics, ControlSequenceNumpy
from simple_worm_experiments.util import dimensionless_MP, default_solver, get_solver, save_output, load_data

from mp_progress_logger import FWException

class ContractionRelaxationExperiment():

    def __init__(self, N, dt, solver = None, quiet = False):
        
        
        if solver is None:
            solver = default_solver()
            
        self.solver = solver
        self.worm = CosseratRod_2(N, dt, self.solver, quiet = quiet)

    def control_sequence(self, parameter):
        '''
        Initialize controls for contraction relation experiment
        
        :param parameter:
        '''

        # TODO: Think about temporal gradual onset for controls        
        N, dt = parameter['N'], parameter['dt'] 
        smo, gmo_t = parameter['smo'], parameter['gmo_t']              
        T0, T = parameter['T0'], parameter['T']
                
        n = round(T/dt)
                                                                                                               
        k, sig = parameter['k0'], parameter['sig0'] 

        if k.size == 3:
            
            k_arr = np.zeros((3, N))
                                    
            k_arr[0, :] = k[0]
            k_arr[1, :] = k[1]
            k_arr[2, :] = k[2]
        else:
            k_arr = k 
                                        
        if sig.size == 3:
            
            sig_arr = np.zeros((3, N))
            
            sig_arr[0, :] = sig[0]
            sig_arr[1, :] = sig[1]
            sig_arr[2, :] = sig[2]
        else:
            sig_arr = sig
                  
        # Gradual muscle unset is achieved by multiplying preferred controls 
        # with sigmoids at the head and the tale        
        if smo: # Body position dependend controls
                                                
            a  = parameter['a_smo']
            du = parameter['du_smo']

            s = np.linspace(0, 1, N)
            
            # sigmoid tale                                    
            st = 1 / (1 + np.exp(-a*(s - du)))
            # sigmoid head
            sh = 1 / (1 + np.exp( a*(s - 1 + du)))
            
            k_arr = st[None, :]*sh[None, :]*k_arr
            sig_arr = st[None, :]*sh[None, :]*sig_arr
                                                       
        k_arr = np.repeat(k_arr[None, :, :], n, axis = 0)                    
        sig_arr = np.repeat(sig_arr[None, :, :], n, axis = 0)
        
        # Gradual temporal muscle unset at the start of the active 
        # deformation and at the start of the passive relaxation
        # is achieved by multiplying the preferred controls 
        # with two sigmoids         
        if gmo_t:
            
            t_arr = np.arange(dt, T+0.1*dt, dt)
                                                                      
            tau_on  = parameter['tau_on']
            Dt_on = parameter['Dt_on']
            
            tau_off  = parameter['tau_off']
            Dt_off = parameter['Dt_off']

            s1 = 1 / (1 + np.exp(-(t_arr - Dt_on) / tau_on))
            s2 = 1 / (1 + np.exp( (t_arr - T0 - Dt_off) / tau_off))
                        
            k_arr = k_arr*s1[:, None, None]*s2[:, None, None]
            sig_arr = sig_arr*s1[:, None, None]*s2[:, None, None]
                                                      
        # Convert numpy array to fenics Function
        CS = [ControlsFenics(v2f(k, Function(self.worm.function_spaces['Omega']), W = self.worm.function_spaces['Omega']), 
                             v2f(sig, Function(self.worm.function_spaces['sigma']), W = self.worm.function_spaces['Omega'])) 
                             for (k,sig) in zip(k_arr, sig_arr)]
                                                                        
        CS = ControlSequenceFenics(CS)
        CS.rod = self.worm
            
        return CS  
    
    def simulate_contraction_relaxation(self, 
                                        parameter, 
                                        pbar = None,
                                        logger = None):            


        T = parameter['T']
        dt_report = parameter['dt_report']
        N_report = parameter['N_report']
                                                                                                                                                       
        MP = dimensionless_MP(parameter)
        CS = self.control_sequence(parameter) 
        F0=None
        
        FS, e = self.worm.solve(T, MP, CS, F0, pbar, logger, dt_report, N_report) 

        CS = CS.to_numpy(dt_report = dt_report, N_report = N_report)
                                  
        return FS, CS, MP, e    
    

def wrap_contraction_relaxation(_input, 
                                pbar,
                                logger,
                                task_number,                             
                                output_dir,  
                                overwrite  = False, 
                                save_keys = None,                              
                             ):
    '''
    Saves simulations results to file
    
    
    :param _input (tuple): parameter dictionary and hash
    :param pbar (tqdm.tqdm): progressbar
    :param logger (logging.Logger): Logger
    :param task_number (int):
    :param output_dir (str): result directory
    :param overwrite (bool): If true, exisiting files are overwritten
    :param save_keys (list): List of attributes which will saved to the result file.
    If None, then all files get saved.
    '''

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
    
    N = parameter['N']
    dt = parameter['dt']
        
    CRE = ContractionRelaxationExperiment(N, dt, solver = get_solver(parameter), quiet = True)
                        
    FS, CS, MP, e = CRE.simulate_contraction_relaxation(parameter, pbar, logger)

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
        