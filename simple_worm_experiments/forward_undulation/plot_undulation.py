'''
Created on 3 Oct 2021

@author: lukas
'''
# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
# Local imports
from simple_worm_experiments.util import get_point_trajectory, compute_com, comp_mean_com_velocity, comp_midline_error
from simple_worm.util_experiments import color_list_from_values

#===============================================================================

fig_path = "../../fig/py/forward_undulation/"

lfz = 16

color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
color_list = 10 * color_list 

cmap = get_cmap('plasma')
    
def plot_single_point_trajactory(ax, FS_arr, point = 'midpoint', undu_plane = 'xy', color_list = color_list):
    """Plot single point trajectory"""

    for i, FS in enumerate(FS_arr):
        
        x_1, x_2 = get_point_trajectory(FS, point = point, undu_plane = undu_plane)
                        
        ax.plot(x_1, x_2, ls = '--', c = color_list[i])
    
    ax.set_xlabel('$x$', fontsize = lfz)
    ax.set_ylabel('$y$', fontsize = lfz)
    
    return

def plot_center_of_mass_trajectory(ax, FS_arr, undu_plane = 'xy', color_list = color_list):
    """Plot center of mass trajectory"""
        
    for i, FS in enumerate(FS_arr):
        
        dt = FS.times[1] - FS.times[0]

        x_com, _, _ = compute_com(FS, dt)
        
        if undu_plane == 'xy':    
            x_com_1 = x_com[:, 0]
            x_com_2 = x_com[:, 1]        
        elif undu_plane == 'xz':        
            x_com_1 = x_com[:, 0]
            x_com_2 = x_com[:, 2]
        elif undu_plane == 'yz':
            x_com_1 = x_com[:, 1]
            x_com_2 = x_com[:, 2]
                        
        ax.plot(x_com_1, x_com_2, ls = '-', c = color_list[i])
    
    return

def plot_center_of_mass_velocity(ax, FS_arr, color_list = color_list):
    """Plot trajectory of center of mass"""
    
    for i, FS in enumerate(FS_arr):
        
        dt = FS.times[1] - FS.times[0]

        _ , v_com, _ = compute_com(FS, dt)
                                
        ax.plot(FS.times, v_com, ls = '-', c = color_list[i])
    
    ax.set_xlabel('time', fontsize = lfz)
    ax.set_ylabel(r'$U(t)$', fontsize = lfz)
        
    return

def plot_mean_center_of_mass_velocity(ax, FS_arr, key, val_arr, Delta_T = 0.3, semilogx = False):

    
    U_arr = []
    
    for i, FS in enumerate(FS_arr):
        
        if not np.isscalar(Delta_T):
            Delta_T_i = Delta_T[i]
        else:
            Delta_T_i = Delta_T
                
        U = comp_mean_com_velocity(FS, Delta_T = Delta_T_i)
        U_arr.append(U)

    if semilogx:
        ax.set_xscale('log')
    
    c = color_list_from_values(val_arr, cmap, log=semilogx)
    
    sc = ax.scatter(val_arr, 
                    U_arr, 
                    marker = 'o',
                    c = c,
                    cmap = cmap, 
                    label = f'{key}={round(val_arr[i],2)}')
        
    #cbar = plt.colorbar(sc)
    #cbar.set_label(key, rotation=270)
    
    ax.set_xlabel(f'{key}', fontsize = lfz)
    ax.set_ylabel(r'$\bar{U}$', fontsize = lfz)
    
    return c

def plot_midline_error(ax, FS_arr, FS_0):

    x_err_mat = comp_midline_error(FS_arr, FS_0)

    t = FS_0.times

    for i, x_err in enumerate(x_err_mat.T):
        
        ax.plot(t, x_err, c = color_list[i])
        
    return
        
def plot_trajectory(FS, parameter, undu_plane = 'xy'):
    
    x1, x2 = get_point_trajectory(FS, point = 'midpoint', undu_plane = undu_plane)

    x_com, v_com, _ = compute_com(FS, parameter['dt'])

    Delta_T = 1.0/parameter['f']

    U = comp_mean_com_velocity(FS, Delta_T = Delta_T)

    # xy-plane
    x_com_1 = x_com[:, 0]
    x_com_2 = x_com[:, 1]

    plt.figure()
    gs = plt.GridSpec(2,1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    ax0.plot(x1, x2, c = 'b', label = 'Midpoint')
    ax0.plot(x_com_1, x_com_2, c = 'r', label = 'COM')
    ax0.set_xlabel(u'$x$', fontsize = 18)
    ax0.set_ylabel(u'$y$', fontsize = 18)    
    ax0.legend(fontsize = 12)
            
    #TODO
    t_arr = FS.times
    
    ax1.plot(t_arr, v_com, c = 'k')
    ax1.plot([t_arr[0], t_arr[-1]], [U,U], c = 'r', ls = '--')
    ax1.set_ylabel(u'$U$', fontsize = 18)
    ax1.set_xlabel(u't', fontsize = 18)

    plt.tight_layout()
             
    return


#------------------------------------------------------------------------------ 
#
def get_FS_arr(parameter, key, val_arr):

    FS_arr = []
    
    for val in val_arr:
    
        # load data
        if type(key) == list:
            for k,v in zip(key, val):            
                parameter[k] = v
        elif type(key) == str:
            parameter[key] = val
                
        data = load_data('undulation_', data_path, parameter)
        FS_arr.append(data['FS'])
    
    return FS_arr


def plot_all(FS, Delta_T = 0.3):
        

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches        
    fig = plt.figure(figsize = (px*1080, px*1920))
    gs = plt.GridSpec(3, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
        
    plot_single_point_trajactory(ax0, FS, point = 'midpoint', color_list = color_list)
    plot_center_of_mass_trajectory(ax0, FS, color_list = color_list) 
    plot_center_of_mass_velocity(ax1, FS_arr, color_list)    

    

def plot_batch(FS_arr, key, val_arr, semilogx = False):
    """Plot a bunch of stuff"""

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches        
    fig = plt.figure(figsize = (px*1080, px*1920))
    gs = plt.GridSpec(3, 1, hspace = 0.3)
        
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    
        
    if semilogx == True:
        # Zero has to be last value in val_arr
        if val_arr[-1] == 0:
            exp1 = np.log10(val_arr[-2])
            exp2 = np.log10(val_arr[-3])
            Delta_exp = exp1 - exp2
            exp = exp1 - Delta_exp
            val_arr[-1] = 10**exp
    
    color_list = color_list_from_values(val_arr, cmap=cmap, log = semilogx)
                                                     
    plot_single_point_trajactory(ax0, FS_arr, point = 'midpoint', color_list = color_list)
    plot_center_of_mass_trajectory(ax0, FS_arr, color_list = color_list) 
    plot_center_of_mass_velocity(ax1, FS_arr, color_list)    
    plot_mean_center_of_mass_velocity(ax2, FS_arr, key, val_arr, Delta_T = 0.3, semilogx = semilogx)
           
    #plt.tight_layout()
                
    return fig


        
        
        
        
        

