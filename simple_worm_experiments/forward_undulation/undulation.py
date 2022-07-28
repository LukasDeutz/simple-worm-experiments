
# Build-in imports
import copy
import itertools as it
from os.path import isfile
from multiprocessing import Pool
import json
import traceback

# Third party imports
from fenics import Expression, Function
import numpy as np
import pickle 

# Local imports
from simple_worm.rod.cosserat_rod_2_test import CosseratRod_2
from simple_worm.controls.controls import ControlsFenics 
from simple_worm.controls.control_sequence import ControlSequenceFenics
from simple_worm.model_parameters import PhysicalParameters, ModelParameters, PP_KEYS, ModelParametersFenics
from simple_worm_experiments.util import dict_hash, dimensionless_MP, default_solver

data_path = "../../data/forward_undulation/"

KINEMATIC_KEYS = ['lam', 'C_t', 'K']
        
#===============================================================================
# Simulate undulation

class ForwardUndulation():
    
    def __init__(self, N, dt, solver = None):
        
        
        if solver is None:
            solver = default_solver()
            
        self.solver = solver
        self.worm = CosseratRod_2(N, dt, self.solver)
                            
    def undulation_control_sequence(self, T, A, lam, f, smooth_muscle_onset = False):
        
        w = 2*np.pi*f
        k = 2*np.pi/lam
            
        n = int(T/self.worm.dt)        
        t_arr = np.linspace(0, T, n)
        
        sigma_expr = Expression(('0', '0', '0'), degree = 1)    
        sigma = Function(self.worm.function_spaces['sigma'])
        sigma.assign(sigma_expr)
        
        if smooth_muscle_onset:
        
            a = 150
            du = 0.05
        
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
        CS = self.undulation_control_sequence(T, parameter['A'], parameter['lam'], parameter['f'], parameter['smooth_muscle_onset']) #parameter['smooth_onset'])                    
        
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

    fn = 'forward_undulation_' + _hash 

    if not overwrite:    
        if isfile(data_path + 'simulations/' + fn + '.dat'):
            print('File already exists')
            return
    
    FU = ForwardUndulation(parameter['N'], parameter['dt'])
                
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
        
    output = {}

    output['parameter'] = parameter
    output['MP'] = MP
        
    if save == 'all':    
        
        output['FS'] = FS        
        output['CS'] = CS.to_numpy()
    
    elif save == 'min':
                
        output['x'] = FS.x
        output['x_t'] = FS.x_t
        output['k'] = FS.Omega
        output['k0'] = CS.Omega
        output['t'] = np.array(FS.times)
                                                            
    pickle.dump(output, open(data_path + '/simulations/' + fn + '.dat', 'wb'))
                    
    return

def simulate_parallel(PG):
    
    for param in PG.param_arr:
    
        pass
    


#===============================================================================
# Analyse simulation results 

def compute_com(FS, dt):
    '''Compute the center of mass and its velocity as a function of time'''
        
    x_com = np.mean(FS.x, axis = 2)
        
    v_com_vec = np.gradient(x_com, dt, axis=0, edge_order=1)    
    v_com = np.sqrt(np.sum(v_com_vec**2, axis = 1))

    return x_com, v_com, v_com_vec 

def get_point_trajectory(FS, point = 'head', undu_plane = 'xy'):
    
    # check if motion is planar
    tol = 1e-3    
    
    if point == 'head':
        point_idx = 0
    elif point == 'tale':
        point_idx = -1
    elif point == 'midpoint':
        point_idx = int(FS.x.shape[2]/2)
        
    # Head positions
    X = FS.x[:, :, point_idx]  
        
    if undu_plane == 'xy':    
        assert np.all(FS.x[:,2,:] < tol), 'locomotion is not planar'
        X_1 = X[:, 0]
        X_2 = X[:, 1]    
    elif undu_plane == 'xz':        
        assert np.all(FS.x[:,1,:] < tol), 'locomotion is not planar'
        X_1 = X[:, 0]
        X_2 = X[:, 2]        
    elif undu_plane == 'yz':
        assert np.all(FS.x[:,0,:] < tol), 'locomotion is not planar'
        X_1 = X[:, 1]
        X_2 = X[:, 2]
    elif undu_plane == 'xyz':
        return X 
    
    return X_1, X_2

def comp_mean_com_velocity(FS, Delta_T = 0.3):
    
    t = np.array(FS.times)
    dt = t[1] - t[0] 

    _, v_com, _ = compute_com(FS, dt)
            
    idx_arr = t >= Delta_T
    
    t = t[idx_arr]
    v = v_com[idx_arr]
    
    U = np.mean(v)
    
    return U 
    
def comp_midline_error(FS_arr, FS_0):
    """Compute error between midlines assuming identical N and dt"""

    N = np.size(FS_0.x, 2)
    du = 1/(N-1)

    x_err_mat = np.zeros((np.size(FS_0.x, 0), len(FS_arr))) 

    for i, FS in enumerate(FS_arr):
                                
        # Euclidean norm
        x_err = np.sum((FS.x - FS_0.x)**2, axis = 1)
        # Integral over midline
        x_err = np.trapz(x_err, axis = 1, dx = du)
        
        x_err_mat[:, i] = x_err
        
    return x_err_mat
        
def compute_work():
    
    # compute work done by the model
    
    # potential energy
    # kinetic energy 
    
    pass
    
    
    

