'''
Created on 28 Jun 2022

@author: lukas
'''


# Build-in importsO
import itertools as it
from multiprocessing import Pool

# Third party imports
import numpy as np
from matplotlib.cm import get_cmap
import pickle
from parameter_scan.util import dict_hash

# Local imports
from simple_worm.model_parameters import PhysicalParameters, ModelParameters, PP_KEYS, MP_KEYS, MODP_KEYS
from simple_worm.rod.solver import Solver

#------------------------------------------------------------------------------ 
# Simulation 

def simulate_batch_parallel(N_worker, data_path, func, PG, overwrite = False, save = 'all', _try = True):
    '''
    Simulate undulation experiment for all parameter dictionaries in grid
    
    :param N_worker: Number of workers
    :param data_path: data_path
    :param func: simulation function
    :param function: Function to be parallelized 
    :param PG: Parameter grid
    :param overwrite: Overwrite already existing simulation results
    :param save: Output   
    :param _try: If true, then the simulation is run I try block
    '''        
        
    # Simulate using N cores in parallel 
    with Pool(N_worker) as pool:
                
        pool.starmap(func, 
                     zip(PG.param_arr,
                     it.repeat(data_path),                     
                     PG.hash_arr, 
                     it.repeat(overwrite), 
                     it.repeat(save),
                     it.repeat(_try)))
        
    return

def load_data(prefix, data_path, parameter, encoding = 'rb'):
    
    filename = prefix + dict_hash(parameter)  
    
    data = pickle.load(open(data_path + filename + '.dat', encoding))
    
    return data
       
#===============================================================================
# Post processing

def align_FS_arr(FS_arr, N_arr, dt_arr):
    """Aligns times for all FS in FS_arr"""
            
    N_min = np.min(N_arr)
    dt_max = np.max(dt_arr)
    
    aligned_FS_arr = []
            
    for i, (N, dt) in enumerate(it.product(N_arr, dt_arr)):
                        
        skip_t = round(dt_max/dt)
        skip_N = round(N/N_min)
        aligned_FS_arr.append(FS_arr[i].clone_skip(skip_t = skip_t, skip_N = skip_N))
        
    return aligned_FS_arr, N_min, dt_max     

def frame_trafo(mat, FS, trafo = 'l2b'):
    
    e1_mat = FS.e1
    e2_mat = FS.e2
    e3_mat = FS.e3
                
    mat_bar = np.zeros_like(mat)
                                                  
    for i, (arr, e1_arr, e2_arr, e3_arr) in enumerate(zip(mat, e1_mat, e2_mat, e3_mat)):
        for j, (x, e1, e2, e3) in enumerate(zip(arr.T, e1_arr.T, e2_arr.T, e3_arr.T)):
            
            Q = np.vstack((e1, e2, e3)).T
            
            # Transposed of Q transforms from lab to body frame
            if trafo == 'l2b':
                Q = Q.T
        
            mat_bar[i, :, j] = np.matmul(Q, x)
    
    return mat_bar
        
#===============================================================================
# Simulation parameter

def default_solver():

    solver = Solver(linearization_method = 'picard_iteration', 
                    finite_difference_order={1:2, 2:2},
                    deepcopy = True)
    
    return solver
            
def default_parameter(parameter = None):
    
    # Simulation parameter
    N = 129
    dt = 0.01
    
    # Model parameter
    external_force = ['sbt']
    rft = 'Whang'
    use_inertia = False
        
    # Geometric parameter    
    L0 = 1130 * 1e-6
    r_max = 32 * 1e-6
    
    # Material parameter
    E = 1e5
    G = E / (2 * (1 + 0.5))
    eta = 1e-2 * E 
    nu = 1e-2 * G
    
    # Fluid 
    mu = 1e-3
                
    if parameter is None:            
        parameter = {}

    parameter['external_force'] = external_force
    parameter['use_inertia'] = use_inertia
    parameter['rft'] = rft
    
    parameter['N']  = N
    parameter['dt'] = dt
    parameter['L0'] = L0
    parameter['r_max'] = r_max
    parameter['E'] = E
    parameter['G'] = G
    parameter['eta'] = eta
    parameter['nu'] = nu
    parameter['mu'] = mu
                
    return parameter

def dimensionless_MP(parameter):
            
    PP_param  = {key: parameter[key] for key in PP_KEYS if key in parameter.keys()}
    MP_param  = {key: parameter[key] for key in MP_KEYS if key in parameter.keys()}
    MOD_param = {key: parameter[key] for key in MODP_KEYS if key in parameter.keys()}
                                                                                
    PP = PhysicalParameters(**PP_param)        
                        
    MP = ModelParameters(physical_parameters = PP, **MP_param, **MOD_param)
     
    if 'gamma' in MP_param:
        MP.gamma = MP_param['gamma']
    if 'c_t' in MP_param:
        MP.c_t = MP_param['c_t']
    if 'c_n' in MP_param:
        MP.c_n = MP_param['c_n']
                   
    return MP

#===============================================================================
# Analyse simulation results 

def compute_com(FS, dt):
    '''Compute the center of mass and its velocity as a function of time'''
        
    x_com = np.mean(FS.x, axis = 2)
        
    v_com_vec = np.gradient(x_com, dt, axis=0, edge_order=1)    
    v_com = np.sqrt(np.sum(v_com_vec**2, axis = 1))

    return x_com, v_com, v_com_vec 

def comp_mean_com_velocity(FS, Delta_T = 0.0):
    
    t = np.array(FS.times)
    dt = t[1] - t[0] 

    _, v_com, _ = compute_com(FS, dt)
                
    v = v_com[t >= Delta_T]
    
    U = np.mean(v)
    
    return U 

def get_point_trajectory(FS, point = 'head', undu_plane = 'xy'):
    
    # check if motion is planar
    tol = 1e-3    
    
    if point == 'head':
        point_idx = 0
    elif point == 'tale':
        point_idx = -1
    elif point == 'midpoint':
        point_idx = int(FS.x.shape[2]/2)
        
    # Head/tale or midpoint coordinates
    X = FS.x[:, :, point_idx]  

    if undu_plane == 'xy':    
        assert np.all(FS.x[:,2,:] < tol), 'locomotion is not planar'
        X_1 = X[:, 0]
        X_2 = X[:, 1]    
        return X_1, X_2
    elif undu_plane == 'xz':        
        assert np.all(FS.x[:,1,:] < tol), 'locomotion is not planar'
        X_1 = X[:, 0]
        X_2 = X[:, 2]        
        return X_1, X_2        
    elif undu_plane == 'yz':
        assert np.all(FS.x[:,0,:] < tol), 'locomotion is not planar'
        X_1 = X[:, 1]
        X_2 = X[:, 2]
        return X_1, X_2        
        
    return X
    
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

#------------------------------------------------------------------------------ 
# Plotting
    
def color_list_from_values(val_arr, cmap = None, log = False):
    if cmap is None:        
        cmap = get_cmap('plasma')
    
    # use log-scale if all values are positive
    if log:
        if (np.array(val_arr) >= 0).sum() == len(val_arr): 
            val_arr = np.log(val_arr)
            
    v_min = np.min(val_arr)
    v_max = np.max(val_arr)
    
    val_arr = (val_arr - v_min) / (v_max - v_min)
    
    color_list = cmap(val_arr)
    
    return color_list


    





