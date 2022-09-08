
# Build-in imports
import copy
from os.path import isfile, join

# Third party imports
from fenics import Expression, Function
import numpy as np

# Local imports
from simple_worm.rod.cosserat_rod_2_test import CosseratRod_2
from simple_worm.controls.controls import ControlsFenics 
from simple_worm.controls.control_sequence import ControlSequenceFenics
from simple_worm_experiments.util import dimensionless_MP, default_solver, get_solver, save_output

from mp_progress_logger import FWException

data_path = "../../data/forward_undulation/"

KINEMATIC_KEYS = ['lam', 'C_t', 'K']
        
#===============================================================================
# Simulate undulation

class ForwardUndulationExperiment():
    
    def __init__(self, N, dt, solver = None, quiet = False):
        
        
        if solver is None:
            solver = default_solver()
            
        self.solver = solver
        self.worm = CosseratRod_2(N, dt, self.solver, quiet = quiet)
                            
    def undulation_control_sequence(self, parameter):
        
        T, smo = parameter['T'], parameter['smo']
        A, lam, f = parameter['A'], parameter['lam'], parameter['f']        
        
        w = 2*np.pi*f
        k = 2*np.pi/lam
            
        n = int(T/self.worm.dt)        
        t_arr = np.linspace(0, T, n)
        
        sigma_expr = Expression(('0', '0', '0'), degree = 1)    
        sigma = Function(self.worm.function_spaces['sigma'])
        sigma.assign(sigma_expr)
        
        if smo: # Smooth muscle onset
        
            a  = parameter['a_smo']
            du = parameter['du_smo']
        
            st = Expression('1/(1 + exp(-a*(x[0] - du)))', degree = 1, a=a, du = du) # sigmoid tale 
            sh = Expression('1/(1 + exp( a*(x[0] - 1 + du)))', degree = 1, a=a, du = du) # sigmoid head
        
            Omega_expr = Expression(("st*sh*A*sin(k*x[0] - w*t)", 
                                     "0",
                                     "0"), 
                                     degree=1,
                                     t = 0,
                                     A = A,
                                     k = k,
                                     w = w,
                                     st = st,
                                     sh = sh)  

        else:
            
            Omega_expr = Expression(("A*sin(k*x[0] - w*t)", 
                                     "0",
                                     "0"), 
                                     degree=1,
                                     t = 0,
                                     A = A,
                                     k = k,
                                     w = w)  
            
                                
        CS = []
            
        for t in t_arr:
            
            Omega_expr.t = t
            Omega = Function(self.worm.function_spaces['Omega'])        
            Omega.assign(Omega_expr)
    
            
            C = ControlsFenics(Omega, sigma)
            CS.append(C)
                        
        CS = ControlSequenceFenics(CS)
        CS.rod = self.worm
            
        return CS  
                                  
    def initial_posture(self, MP, C):
        
        # Relax to initial posture     
        T = 2.0
        
        n = int(T/self.worm.dt) 
            
        #TODO: Use rft instead of linear drag 
        MP.external_force = 'linear_drag'
        MP.K_t = np.identity(3)
        
        CS = ControlSequenceFenics(C, n_timesteps = n)
        
        FS = self.worm.solve(T, MP = MP.to_fenics(), CS = CS)
          
        return FS[-1]        
        
    def simulate_undulation(self, 
                            parameter, 
                            pbar = None,
                            logger = None,
                            init_posture = False):            


        T = parameter['T']
                                                                                                                                    
        MP = dimensionless_MP(parameter)
        CS = self.undulation_control_sequence(parameter) 
        
        if init_posture:    
            #TODO:
            F0 = self.initial_posture(copy.deepcopy(MP), CS[0])
        else:
            F0 = None
                    
        FS, e = self.worm.solve(T, MP, CS, F0, pbar, logger) 

        CS = CS.to_numpy()
                                  
        return FS, CS, MP, e

#------------------------------------------------------------------------------ 
# 

def wrap_simulate_undulation(_input, 
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
                    
    N = parameter['N']
    dt = parameter['dt']
        
    FU = ForwardUndulationExperiment(N, dt, solver = get_solver(parameter), quiet = True)
    
                    
    FS, CS, MP, e = FU.simulate_undulation(parameter, pbar, logger)

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
        raise FWException(FS.pic, parameter['T'], parameter['dt'], FS.times[-1]) from e
        
    # If simulation has finished succesfully then we return the relevant results 
    # for the logger
    result = {}    
    result['pic'] = FS.pic
    
    return result
    
                

    


    
    
    

