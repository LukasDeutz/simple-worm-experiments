
# Build-in imports
import copy
from os.path import isfile
import json
import traceback

# Third party imports
from fenics import Expression, Function
import numpy as np

# Local imports
from simple_worm.rod.cosserat_rod_2_test import CosseratRod_2
from simple_worm.controls.controls import ControlsFenics 
from simple_worm.controls.control_sequence import ControlSequenceFenics
from simple_worm_experiments.util import dimensionless_MP, default_solver, get_solver, save_output

data_path = "../../data/forward_undulation/"

KINEMATIC_KEYS = ['lam', 'C_t', 'K']
        
#===============================================================================
# Simulate undulation

class ForwardUndulationExperiment():
    
    def __init__(self, N, dt, solver = None):
        
        
        if solver is None:
            solver = default_solver()
            
        self.solver = solver
        self.worm = CosseratRod_2(N, dt, self.solver)
                            
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
                            init_posture = False):            


        T = parameter['T']
                                                                                                                                    
        MP = dimensionless_MP(parameter)
        CS = self.undulation_control_sequence(parameter) 
        
        if init_posture:    
            #TODO:
            F0 = self.initial_posture(copy.deepcopy(MP), CS[0])
        else:
            F0 = None
                    
        FS = self.worm.solve(T, CS=CS, MP = MP, F0 = F0) 
                                  
        return FS, CS, MP

#------------------------------------------------------------------------------ 
# 

def wrap_simulate_undulation(parameter, data_path, _hash, overwrite = False, save = 'min', _try = False):

    fn = _hash 

    if not overwrite:    
        if isfile(data_path + 'simulations/' + fn + '.dat'):
            print('File already exists')
            return
        
    FU = ForwardUndulationExperiment(parameter['N'], parameter['dt'], solver = get_solver(parameter))
                
    if not _try:
        FS, CS, MP = FU.simulate_undulation(parameter)
    
    else:    
        try:        
            FS, CS, MP = FU.simulate_undulation(parameter)
        except Exception as e:
            traceback.print_exception(e)                
            with open(data_path + '/errors/' + 'error_report_' + _hash + '.json', 'w') as f:        
                json.dump(parameter, f, indent=4)    
            return
        
    save_output(data_path, fn, FS, MP, CS, parameter, save)

    return


    
    
    

