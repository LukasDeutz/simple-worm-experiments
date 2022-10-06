'''
Created on 6 Oct 2022

@author: lukas
'''
import numpy as np

def undulation_indexes(parameter, t):
    '''
    Returns indexes of timesteps befor and after the 
    turn maneuver 
    
    :param parameter (dict): parameter dictionary 
    :param t (np.array): times
    '''
    
    f = parameter['f0']
    t0 = parameter['t0']
    Dt = parameter['Delta_t']

    T = 1/f
            
    idx1 = (t > T) & (t < t0)     
    idx2 = t > t0
    idx3 = t > (t0 + Dt + T)  
    
    return idx1, idx2, idx3

def compute_turning_angle(x, t, parameter):
    '''
    Computes turning angle of the roll maneuver
    
    :param x:
    :param t:
    :param parameter:
    '''
                          
    idx1, _ , idx3 = undulation_indexes(parameter, t)
        
    # turn angle
            
    x1 = x[idx1]
    x2 = x[idx3]
    
    x1_com = np.mean(x1, axis = 2)        
    t1_com = np.diff(x1_com, axis = 0)        
    avg_t1_com = np.mean(t1_com, axis = 0)        
    avg_t1_com = avg_t1_com / np.linalg.norm(avg_t1_com)
    
    x2_com = np.mean(x2, axis = 2)        
    t2_com = np.diff(x2_com, axis = 0)        
    avg_t2_com = np.mean(t2_com, axis = 0)
    avg_t2_com = avg_t2_com / np.linalg.norm(avg_t2_com)
    
    #TODO: Only works if the undulation plane is the xy-plane 
    
    # To caculate the turning anlge, we use polar coordinates.
    # The average direction of travel before the roll
    # maneuver avg_t1_com, points in positive x direction.
    # The y-axis is then chosen to be orthogonal to the x-axis
    # such that left turns have negative angles and right turns
    # have positive angles     
    e_x = avg_t1_com[0:2]
    e_y = np.array([e_x[1], -e_x[0]])
              
    x = np.sum(e_x * avg_t2_com[0:2])
    y = np.sum(e_y * avg_t2_com[0:2])
                                    
    phi = np.arctan2(y, x)
                                     
    return phi


