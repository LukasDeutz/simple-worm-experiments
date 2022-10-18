'''
Created on 28 Jun 2022

@author: lukas
'''
# Build-in imports
import itertools as it
from multiprocessing import Pool
import numpy as np
from scipy.spatial import distance_matrix

# Third party imports
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pickle
from parameter_scan.util import dict_hash, load_file, load_file_grid

# Local imports
from simple_worm.model_parameters import PhysicalParameters, ModelParameters, PP_KEYS, MP_KEYS, MODP_KEYS
from simple_worm.rod.solver import Solver, SOLVER_KEYS

#------------------------------------------------------------------------------ 
# Simulation 

def simulate_batch_parallel(N_worker, 
                            data_path, 
                            func, 
                            PG, 
                            overwrite = False, 
                            save = 'all', 
                            _try = True, 
                            quiet = False):
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
    :param quiet: If true, solver does not print progress
    '''        
        
    # Simulate using N cores in parallel 
    with Pool(N_worker) as pool:
                
        pool.starmap(func, 
                     zip(PG.param_arr,
                     it.repeat(data_path),                     
                     PG.hash_arr, 
                     it.repeat(overwrite), 
                     it.repeat(save),
                     it.repeat(_try),
                     it.repeat(quiet)))
        
    return

def load_data(prefix, data_path, parameter, encoding = 'rb'):
    
    filename = prefix + dict_hash(parameter)  
    
    data = pickle.load(open(data_path + filename + '.dat', encoding))
    
    return data

def save_output(filepath, 
                FS, 
                CS, 
                MP, 
                parameter, 
                exit_status = 1,
                save_keys = None):
    '''
    Save simulation results
    
    :param filepath (str):
    :param filename (str):
    :param FS (FrameSequenceNumpy):
    :param CS (ControlSequenceNumpy):
    :param MP (...):
    :param parameter (dict):
    :param exit_status (int): if 0, then the simulation finished succesfully, if 1, error occured  
    :param save_keys (dict):
    '''

    output = {}

    output['parameter'] = parameter
    output['MP'] = MP
    output['exit_status'] = exit_status
        
    if save_keys is None:    
        
        output['FS'] = FS        
        output['CS'] = CS
    
    else:
        for key in save_keys:
        
            output[key] = getattr(FS, key)
                     
    with open(filepath, 'wb') as file:
                                                                            
        pickle.dump(output, file)
    
    return


def get_filename_from_PG_arr(PG_arr):
        
    return '_'.join([PG.filename for PG in PG_arr]) + '.dat'   
    
       
def write_sim_data_to_file(data_path, PG_arr, output_keys, file_path = None, prefix = ''):

    data_dict = {}
                                            
    if file_path is None:        
        file_path = data_path + get_filename_from_PG_arr(PG_arr)
                                            
    for PG in PG_arr:
        
        hash_arr = PG.hash_arr
        data = {key : [] for key in output_keys}
        
        for _hash in hash_arr:

            file = load_file(data_path, _hash, prefix = prefix)
            FS = file['FS']
                            
            for key in output_keys: 
            
                data[key].append(getattr(FS, key))

        data_dict[PG.filename] = data
                                                                        
    with open(file_path, 'wb') as outfile:
        pickle.dump(data_dict, outfile, pickle.HIGHEST_PROTOCOL)                
            
    return file_path

def check_if_pic(data_path, PG):
    '''
    Checks if the picard iteration for all time steps
    and all frame sequences for given parameter grid    
        
    :param data_path: data path of simulations files
    :param PG: Parameter grid
    '''
    
    file_grid = load_file_grid(PG.hash_grid, data_path)
    file_arr = file_grid.flatten()
    
    pic_arr = np.concatenate([file['FS'].pic for file in file_arr])
        
    print(f'PI-solver converged at every time step of every simulation: {np.all(pic_arr.flatten())}')
    
    return np.all(pic_arr.flatten())
                   
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

def frame_trafo(v_mat, FS, trafo = 'l2b'):
    
    e1_mat = FS.e1
    e2_mat = FS.e2
    e3_mat = FS.e3
                
    v_mat_bar = np.zeros_like(v_mat)
                                                  
    for i, (arr, e1_arr, e2_arr, e3_arr) in enumerate(zip(v_mat, e1_mat, e2_mat, e3_mat)):
        for j, (v, e1, e2, e3) in enumerate(zip(arr.T, e1_arr.T, e2_arr.T, e3_arr.T)):
            
            Q = np.vstack((e1, e2, e3))
            
            # Transposed of Q transforms from lab to body frame
            if trafo == 'l2b':
                v_mat_bar[i, :, j] = np.matmul(Q, v)
            elif trafo == 'b2l':
                v_mat_bar[i, :, j] = np.matmul(Q.T, v)
                    
    return v_mat_bar
                            
#===============================================================================
# Simulation parameter

def get_solver(parameter):

    solver_param = {k: parameter[k] for k in parameter if k in SOLVER_KEYS}

    return Solver(**solver_param)

def default_solver():
    
    return Solver()

           
def default_parameter(parameter = None):
    
    # Simulation parameter
    N = 129
    dt = 0.01
    
    # Model parameter
    external_force = ['sbt']
    rft = 'Whang'
    use_inertia = False
    rc = 'spheroid'
    
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
    parameter['rc'] = rc
        
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

def compute_com(x, dt):
    '''Compute the center of mass and its velocity as a function of time'''
        
    x_com = np.mean(x, axis = 2)
        
    v_com_vec = np.gradient(x_com, dt, axis=0, edge_order=1)    
    v_com = np.sqrt(np.sum(v_com_vec**2, axis = 1))

    return x_com, v_com, v_com_vec 

def comp_mean_com_velocity(x, t, Delta_T = 0.0):
    '''
    Computes mean swimming speed
    
    :param x: centreline coordinates
    :param t: time
    :param Delta_T: crop timepoints t < Delta_T
    '''    
    dt = t[1] - t[0] 

    _, v_com, _ = compute_com(x, dt)
                
    v = v_com[t >= Delta_T]
    
    U = np.mean(v)
    
    return U 

def comp_mean_work(dot_W, t, Delta_T = 0.0):
    '''
    Computes mean work done by the swimmer
    
    :param dot_W (np.array): rate of work array
    :param t (np.array): time array
    :param Delta_T: timepoints t < Delta_T are cropped
    '''
    
    dot_W = dot_W[t >= Delta_T]

    return np.mean(dot_W)
    
def comp_angle_of_attack(x, time, Delta_t = 0):
    '''
    Compute angle of attack
    
    :param x (np.array): centreline coordinates
    :param t (np.array): times 
    :param Delta_t (float): crop times points smaller Delta_t average over time
    '''        
    dt = time[1] - time[0]
    
    _, _, v_com_vec = compute_com(x, dt)
    
    # Average com velocity 
    v_com_avg = np.mean(v_com_vec, axis = 0)
    # Propulsion direction
    e_p = v_com_avg / np.linalg.norm(v_com_avg)

    # Compute tangent   
    # Negative sign makes tangent point from tale to head 
    t = - np.diff(x, axis = 2)
    # Normalize
    abs_tan = np.linalg.norm(t, axis = 1)    
    t = t / abs_tan[:, None, :]
    
    # Turning angle
    # Arccos of scalar product between local tangent and propulsion direction
    dot_t_x_e_p = np.arccos(np.sum(t * e_p[None, :, None], axis = 1))
    
    theta = (dot_t_x_e_p)  
    # Think about absolute value here      
    avg_theta = np.mean(np.abs(theta), axis = 1)
        
    # Time average
    t_avg_theta = np.mean(avg_theta[time >= Delta_t])
        
    return theta, avg_theta, t_avg_theta
    
        
def comp_distant_matrix(x, n_skip = None):   
    '''
    Compute distance matrix between centreline gird points
    
    :param x (np.array): centreline coordinates
    :param n_skip (int): n most direct neighbours are skipped
    '''
    N = x.shape[2] # body points
    n = x.shape[0] # number timesteps
    
    x_dist_mat = np.zeros((n, N, N))
                                
    for i, x_t in enumerate(x):

        x_dist_mat_t = distance_matrix(x_t.T, x_t.T)        
        # Set diagonal distances to v 
        
        np.fill_diagonal(x_dist_mat_t, np.nan)
                
            # Set distances to n most direct left and right neightbours to inf
            
        if n_skip is not None:

            for j in range(1, n_skip+1):
                x_dist_mat_t += np.diag(np.nan*np.ones(N-j),  j)
                x_dist_mat_t += np.diag(np.nan*np.ones(N-j), -j)
        
        x_dist_mat[i, :, :] = x_dist_mat_t
                            
    return x_dist_mat

def comp_minimum_distant(x, n_skip):

    x_dist_mat = comp_distant_matrix(x, n_skip)
    
    x_min_dist_arr = np.nanmin(x_dist_mat, axis = 2)
    x_min_dist = np.nanmin(x_min_dist_arr, axis = 1)

    return x_min_dist, x_min_dist_arr

def comp_average_distant(x):
         
    x_dist_mat = comp_distant_matrix(x)
         
    x_avg_dist_arr = np.nanmean(x_dist_mat, axis = 2)
    x_avg_dist = np.nanmean(x_avg_dist_arr, axis = 1)

    return x_avg_dist, x_avg_dist_arr
                                  
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
    
def color_list_from_values(val_arr, map_name = None, log = False):

    if map_name is None:    
        cmap = cm.get_cmap('plasma')    
    
    # use log-scale if all values are positive
    if log:
        if (np.array(val_arr) >= 0).sum() == len(val_arr): 
            val_arr = np.log(val_arr)
            
    v_min = np.min(val_arr)
    v_max = np.max(val_arr)
    
    val_arr = (val_arr - v_min) / (v_max - v_min)
    
    color_list = cmap(val_arr)
    
    return color_list

def color_map_from_values(c_arr, log = True, map_name = None):
    
    if map_name is None:    
        cmap = cm.get_cmap('plasma')

    if log:
        assert np.all(c_arr > 0)        
        c_arr = np.log10(c_arr)
            
    normalize = mcolors.Normalize(vmin = np.min(c_arr), vmax = np.max(c_arr))
    s_map = cm.ScalarMappable(norm=normalize, cmap = cmap)
  
    c_list = s_map.to_rgba(c_arr)
  
    return s_map, cmap, c_list 
  
    
    





