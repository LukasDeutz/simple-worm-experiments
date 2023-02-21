'''
Created on 10 Jan 2023

@author: lukas
'''
import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid

class EPP(object):
    '''
    Class to post process and analyse simulation results.
    
    n = number of time points
    N = number of bodypoints        
    '''
    
    #TODO: Powers should be consistently named throughout the project
    rename_powers = {
        'V_dot_k': 'dot_V_k',
        'V_dot_sig': 'dot_V_sig',        
        'D_k': 'dot_D_k',
        'D_sig': 'dot_D_sig',
        'dot_W_F_lin': 'dot_W_F_F',
        'dot_W_F_rot': 'dot_W_F_T',
        'dot_W_M_lin': 'dot_W_M_F',
        'dot_W_M_rot': 'dot_W_M_T'}
        
    energy_names_from_powers = {
        'dot_V_k': 'V_k',
        'dot_V_sig': 'V_sig',
        'dot_D_k': 'D_k',
        'dot_D_sig': 'D_sig',
        'dot_W_F_F': 'W_F_F',
        'dot_W_F_T': 'W_F_T',
        'dot_W_M_F': 'W_M_F',
        'dot_W_M_T': 'W_M_T'}
    
    @staticmethod
    def comp_com(x, dt):
        '''Compute mean centreline coordinates and its velocity as a function of time
                
        :param x (np.ndarray (n x 3 x N)): centreline coordinates
        :param dt (float): timestep                
        '''
            
        x_com = np.mean(x, axis = 2)
            
        v_com_vec = np.gradient(x_com, dt, axis=0, edge_order=1)    
        v_com = np.sqrt(np.sum(v_com_vec**2, axis = 1))
    
        return x_com, v_com, v_com_vec 
        
    @staticmethod
    def comp_mean_com_velocity(x, t, Delta_T = 0.0):
        '''
        Computes mean swimming speed
        
        :param x (np.ndarray (n x 3 x N)): centreline coordinates
        :param t: (np.ndarray (n x 1)): timestamps
        :param Delta_T (float): crop timepoints t < Delta_T
        '''    
        dt = t[1] - t[0] 
    
        _, v_com, _ = EPP.comp_com(x, dt)
                    
        v = v_com[t >= Delta_T]
        
        U = np.mean(v)
        
        return U 
    
    @staticmethod
    def comp_angle_of_attack(x, time, Delta_t = 0):
        '''
        Compute angle of attack
        
        :param x (np.array (n x 3 x N)): centreline coordinates
        :param t (np.array (n x 1)): timestamps
        :param Delta_t (float): crop timepoints t < Delta_T
        '''        
        dt = time[1] - time[0]
        
        _, _, v_com_vec = EPP.compute_com(x, dt)
        
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
    
    @staticmethod
    def centreline_pca(X):
        '''
        Computes principal directions and components
        of the centreline coordinates pooled over time 
        and body position
        
        :param X (np.ndarray (n x 3 x N)): centreline coordinates
        '''
                
        # Reformat into 2D array (3 x (n x N))
        X = np.swapaxes(X, 1, 2)                
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
        
        # Centre coordinates around mean 
        x_avg = X.mean(axis = 0)         
        X -= x_avg[None, :]

        C = np.matmul(X.T, X)            
        lam, w =  np.linalg.eig(C)

        # order from big to small
        idx_arr = lam.argsort()[::-1]
        lam = lam[idx_arr]        
        w = w[:, idx_arr]
                    
        return lam, w, x_avg
            
    @staticmethod
    def project_into_pca_plane(X, 
            w2_ref = None,
            w3_ref = None,
            output_w = False):
        
        _, w, x_avg = EPP.centreline_pca(X)
                
        # Principal directions 2 and 3        
        w2 = w[:, 1]
        w3 = w[:, 2] 
        
        # If reference direction is given 
        # then we choose the sign of eigenvector 
        # such that its angle with respect 
        # direction to the reference is smaller
        # than 90 degrees
        if w2_ref is not None:
            
            if np.dot(w2, w2_ref) < np.dot(w2, w3_ref):
                tmp = w2
                w2 = w3
                w3 = tmp
                        
            dp = np.dot(w2, w2_ref)
            if dp < 0: w2 = -w2
            dp = np.dot(w3, w3_ref)
            if dp < 0: w3 = -w3
                
        # Centre
        X = X - x_avg[None, :, None]
        
        # Project        
        x = np.sum(X*w2[None, :, None], axis = 1) 
        y = np.sum(X*w3[None, :, None], axis = 1)

        if not output_w: return x, y
        else: return x, y, w2, w3
                               
    @staticmethod
    def comp_dv_plane(d2, s_mask = None):
        '''
        Returns normal vector of the dorsal-ventral plane. 
        
        In case of nonplanar postures, the orientation of the 
        ventral-plane varies locally and dorsal-ventral plane
        is defined as the average direction of local 
        d2 body frame vector. 
                
        :param s_mask (np.array): Boolean array to mask spatial dimension
        :return d2_avg (np.array): Average d2 direction
        '''

        if s_mask is not None: d2 = d2[:, :, s_mask]
        
        d2_avg = np.mean(d2, axis = 2)
        d2_avg = d2_avg / np.linalg.norm(d2_avg, axis = 1)[:, None]
        
        return d2_avg 
        
    @staticmethod
    def comp_dv_rot_angle(d2, s_mask = None, d2_ref=None):
        '''
        Computes angle between the normal of the dorsal ventral plane with respect 
        to a reference direction
        
        :param FS (FrameSequenceNumpy): Frame sequence
        :param d2_ref (np.array): Reference direction
        :return phi (np.array): rotation angle        
        '''        
        # Average d2 direction over time
        d2_avg = EPP.comp_dv_plane(d2, s_mask)
        
        # If no reference orientation is given
        # then the average d2 direction of initial 
        # frame is used
        if d2_ref is None:             
            d2_ref = d2[0, :, 0]
                
        phi = np.arccos(np.dot(d2_avg, d2_ref))
        
        return phi
    
    @staticmethod
    def comp_dv_spherical_angle(d2, d123_ref, s_mask = None):
        '''
        Computes the representation of the normal direction of the dorsal ventral 
        plane in spherical coordinates with respect to a given reference frame
                
        :param (FrameSequenceNumpy): Frame sequence
        :param s_mask (np.array): Boolean array to mask spatial dimension
        :param d123_ref (np.array): coordinate axis [d1, d2, d3]
        :return phi (np.array): 
        :return theta (np.array): 
        '''
                        
        d1_ref, d2_ref, d3_ref = d123_ref[0, :], d123_ref[1, :], d123_ref[2, :]
                            
        # Average d2 direction over time
        d2_avg = EPP.comp_dv_plane(d2, s_mask)
        # Project d2_avg onto reference coordinate axes
        z = np.dot(d2_avg, d3_ref)
        y = np.dot(d2_avg, d2_ref)
        x = np.dot(d2_avg, d1_ref)
        
        # Compute angles from cartesian coordinates
        phi = np.arctan2(x,y)
        theta = np.arccos(z)
                
        return phi, theta 

    @staticmethod        
    def comp_roll_frequency(angle, t):
        '''
        Compute roll frequency from roll angle time sequence
        
        :param angle (np.ndarray): roll angle in range [-pi, pi]
        :param t  (np.ndarray): time series
        '''

        # find zero crossings                
        idx_zc = np.abs(np.diff(np.sign(angle))) == 2
                
        # interpret zero crossings of the rotation angle
        # as the start point of a rotation period
        t_start = t[:-1][idx_zc]
        
        # If angle has no zero crossing, set roll frequency to None
        if t_start.size <= 1.0:
            f_avg, f_std = np.NaN, np.NaN
        # Else, approxomate roll frequency as average time period
        # between zero crossings
        else:                    
            f = 0.5 / np.diff(t_start)                        
            f_avg = np.mean(f)
            f_std = np.std(f) 
                          
        return f_avg, f_std
                
    @staticmethod
    def comp_roll_frequency_from_spherical_angle(d2, t, d123_ref, s_mask = None, Dt = None):
        '''
        Compute roll frequency from infinite roll experiment
        
        :param (FrameSequenceNumpy): Frame sequence:
        :param s_mask (np.array): Boolean array to mask spatial dimension:
        :param d2_ref (np.array): Normal direction of dorsal-ventral reference plane
        :param Dt (float): Initiation phase. 
        
        :return f_avg (float): Average roll frequency
        :return f_std (float): Std roll frequency 
        :return phi (np.array): Rotation angle
        
        '''
        
        phi, _ = EPP.comp_dv_spherical_angle(d2, d123_ref, s_mask)
                    
        # Crop initiation phase of the experiment                        
        if Dt is not None:
            idx = t >= Dt
            phi_crop = phi[idx]
            t_crop = t[idx]
        else: 
            phi_crop = phi
            t_crop = t
            
        f_avg, f_std = EPP.comp_roll_frequency(phi_crop, t_crop)
                                                                                                  
        return f_avg, f_std, phi
        
    @staticmethod
    def comp_roll_frequency_from_euler_angle(alpha, t, s_mask = None, Dt = None):
        '''
        Compute roll frequency from continuous roll experiment
        
        n = number of time steps
        N = number of body points 
        
        :param alpha (np.array n x 3 x N): roll angle 
        :param s_mask (np.array N): Boolean array to mask spatial dimension 
        :param Dt (float): Initiation phase. 
        
        :return f_avg (float): Average roll frequency
        :return f_std (float): Std roll frequency         
        '''        
                    
        # Crop initiation phase of the experiment                        
        if Dt is not None:
            idx = t >= Dt
            alpha = alpha[idx]
            t = t[idx]
        if s_mask is not None:
            alpha = alpha[:, s_mask]
                                        
        # map alpha to range -pi to pi
        avg_alpha = alpha.mean(axis = 1)
        avg_alpha = avg_alpha % (2*np.pi) - np.pi        
        
        f_avg, f_std = EPP.comp_roll_frequency(avg_alpha, t)
                                                          
        return f_avg, f_std
                 
    @staticmethod                 
    def powers_from_FS(FS):
        '''
        Returns dicitionary with powers from frame sequence
        
        :param FS (FrameSequenceNumpy): frame sequence
        '''            
        powers = {}
                
        for k, new_k in EPP.rename_powers.items():
            
            powers[new_k] = getattr(FS, k)
        
        return powers

    @staticmethod
    def powers_from_h5(h5, t_start = None, t_end = None):
        '''
        Returns dicitionary with powers from frame hdf5 file
        
        :param h5 (h5py.File): hdf5 file
        '''                                    
        powers = {}

        if t_start is not None or t_end is not None:        
            t = h5['t'][:]
            idx_arr = np.ones(t.size, dtype = bool)                
            if t_start is not None:
                idx_arr = np.logical_and(idx_arr, t >= t_start) 
            if t_end is not None:
                idx_arr = np.logical_and(idx_arr, t <= t_end)             
            t = t[idx_arr]
                
        for k, new_k in EPP.rename_powers.items():
            
            if t_start is not None or t_end is not None:
                powers[new_k] = h5['FS'][k][:][:, idx_arr]
            else:
                powers[new_k] = h5['FS'][k][:]

        if t_start is not None or t_end is not None:
            return powers, t
        else:
            return powers
    
    @staticmethod
    def comp_true_powers(powers):
        '''
        Compute "true" output powers 
        
        :param powers (dict): power dictionary         
        '''
        dot_V = powers['dot_V_k'] + powers['dot_V_sig']
        dot_D = powers['dot_D_k'] + powers['dot_D_sig']                
        dot_W_F = powers['dot_W_F_F'] + powers['dot_W_F_T']
        
        # Increasing the potential energy costs energy
        # and is therefore counted negative
        dot_V = - dot_V
        # Compute "true" dissipation rate        
        dot_E = dot_D + dot_W_F + dot_V
        # If the decrease in potential energy per unit time 
        # is larger than the total dissipation rate then we 
        # set the true disspation to zero 
        idx_arr = dot_E >= 0        
        dot_D[idx_arr] = 0
        dot_W_F[idx_arr] = 0
        dot_V[idx_arr] = -dot_E[idx_arr]
                        
        # If the decrease in potential energy per unit time 
        # is smaller than the total dissipation rate then we 
        # add the released potential energy equally
        # to the internal and fluid dissipation rate
        idx_arr = np.logical_and(dot_V > 0, dot_E < 0)        
        dot_D[idx_arr] += 0.5*dot_V[idx_arr]
        dot_W_F[idx_arr] += 0.5*dot_V[idx_arr]        
        dot_V[idx_arr] = 0        
        idx_arr = dot_D > 0                                                                
        dot_W_F[idx_arr] += dot_D[idx_arr]        
        dot_D[idx_arr] = 0        
        idx_arr = dot_W_F > 0
        dot_D[idx_arr] += dot_W_F[idx_arr]
        dot_W_F[idx_arr] = 0
                                                                                                                  
        return dot_V, dot_D, dot_W_F
                                                                                                            
    @staticmethod
    def comp_energy_from_power(powers, dt):
        '''
        Computes energy which stored as elastic potential energy, dissipated internally or
        into the fluid and mechanical muscle work done from given powers as function of time
        
        :param powers (dict): power dictionary 
        :param dt (float): timestep
        '''
             
        energies = {}
                          
        for key, power in powers.items():
            
            name = EPP.energy_names_from_powers[key]
            energies[name] = cumulative_trapezoid(power, dx=dt, initial=0)       
       
        return energies
       
    @staticmethod
    def comp_tot_energy_from_power(powers, dt):
        '''
        Computes total energy stored as elastic potential, dissipated internally or
        into the fluid and mechanical muscle work done from given powers as function of time
        
        :param powers (dict): power dictionary 
        :param dt (float): timestep
        '''        
        energies = {}
                          
        for key, power in powers.items():
            
            name = EPP.energy_names_from_powers[key]
            energies[name] = cumulative_trapezoid(power, dx=dt, initial=0)       
       
        return energies
        
        
        
        
        
        
        