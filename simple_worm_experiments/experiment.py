'''
Created on 6 Jan 2023

@author: lukas
'''

# Build-in imports
from os.path import isfile, join
import pickle
from fenics import Expression
import numpy as np

# Local imports
from simple_worm.rod.cosserat_rod_2_test import CosseratRod_2
from simple_worm_experiments.util import dimensionless_MP, get_solver, save_output

from mp_progress_logger import FWException
            
class Experiment():      
      
    @staticmethod   
    def sig_m_on(t0, tau):
        
        return lambda t: 1.0 / (1 + np.exp(-( t - t0 ) / tau))
    
    @staticmethod   
    def sig_m_off(t0, tau):

        return lambda t: 1.0 / (1 + np.exp( ( t - t0) / tau))

    @staticmethod
    def sig_m_on_expr(t0, tau):

        return Expression('1 / ( 1 + exp(- ( t - t0) / tau))', 
            degree = 1, 
            t = 0, 
            t0 = t0, 
            tau = tau)

    @staticmethod
    def sig_m_off_expr(t0, tau):

        return Expression('1 / ( 1 + exp( ( t - t0) / tau))', 
            degree = 1, 
            t = 0, 
            t0 = t0, 
            tau = tau)
        
    @staticmethod
    def sig_head_expr(Ds, s0):
        
        return Expression('1 / ( 1 + exp(- (x[0] - s0) / Ds ) )', 
            degree = 1, 
            Ds = Ds, 
            s0 = s0)
    
    @staticmethod
    def sig_tale_expr(Ds, s0):
            
        return Expression('1 / ( 1 + exp(  (x[0] - s0) / Ds) )', 
            degree = 1, 
            Ds = Ds, 
            s0 = s0) 

      
      
def init_worm(parameter):
    '''
    Initiates worm object
    
    :param parameter (dict): parameter
    :returns worm (simple_worm.CosseratRod): worm object    
    '''
        
    N = parameter['N']
    dt = parameter['dt']
    solver = get_solver(parameter)        

    worm = CosseratRod_2(N, dt, solver, quiet=True)
    
    return worm
                                      
def simulate_experiment(worm,
                        parameter,
                        CS, 
                        pbar = None,
                        logger = None,
                        F0 = None):            
    '''
    Simulate experiment defined by control sequence
            
    :param worm (simple_worm.CosseratRod): Worm
    :param parameter (dict): Parmeter
    :param pbar (tqdm.tqdm): Progressbar
    :param logger (logging.Logger): Progress logger
    :param F0 (simple_worm.FrameSequence): Initial frame
    '''
        
    T, dt_report, N_report = parameter['T'], parameter['dt_report'], parameter['N_report']
                                                                                                                                                   
    MP = dimensionless_MP(parameter)
                        
    FS, e = worm.solve(T, MP, CS, F0, pbar, logger, dt_report, N_report) 
            
    CS = CS.to_numpy(dt_report = dt_report, N_report = N_report)
                              
    return FS, CS, MP, e

def wrap_simulate_experiment(_input, 
                             pbar,
                             logger,
                             task_number,                             
                             create_CS,
                             output_dir,  
                             overwrite  = False, 
                             save_keys = None,                              
                             ):
    '''
    Wrapes simulate_experiment function to make it compatible with parameter_scan module. 
        
    :param _input (tuple): Parameter dictionary and hash
    :param create_CS (function): Creates control sequence from parameter
    :param pbar (tqdm.tqdm): Progressbar
    :param logger (logging.Logger): Progress logger
    :param task_number (int): Number of tasks
    :param output_dir (str): Result directory
    :param overwrite (bool): If true, exisiting files are overwritten
    :param save_keys (list): List of attributes which will be saved to the result file. 
        If None, then all attributes get saved.        
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
         
    worm = init_worm(parameter)
                    
    CS = create_CS(worm, parameter)

    FS, CS, MP, e = simulate_experiment(worm, parameter, CS, pbar, logger, None)
                        
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

        
        
