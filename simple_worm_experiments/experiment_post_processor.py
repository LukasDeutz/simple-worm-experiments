'''
Created on 10 Jan 2023

@author: lukas
'''
import numpy as np


class EPP(object):
    '''
    Class to post process and analyse simulation results.
    '''

    def __init__(self):
        '''
        Constructor
        '''
           
    @staticmethod           
    def compute_com(x, dt):
        '''Compute the center of mass and its velocity as a function of time'''
            
        x_com = np.mean(x, axis = 2)
            
        v_com_vec = np.gradient(x_com, dt, axis=0, edge_order=1)    
        v_com = np.sqrt(np.sum(v_com_vec**2, axis = 1))
    
        return x_com, v_com, v_com_vec 
       
        
    @staticmethod
    def comp_dv_plane(d2, s_mask = None):
        '''
        Returns normal vector of the dorsal-ventral plane. 
        
        In case of nonplanar postures, the orientation of the 
        ventral-plane varies locally and dorsal-ventral plane
        is defined as the average direction of local 
        d2 body frame vector. 
                
        :param FS (FrameSequenceNumpy): Frame sequence
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
    def comp_roll_frequency(d2, t, d123_ref, s_mask = None, Dt = None):
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
                                                        
        # find zero crossings                
        idx_zc = np.abs(np.diff(np.sign(phi_crop))) == 2
                
        # interpret zero crossings of the rotation angle
        # as the start point of a rotation period
        t_start = t_crop[:-1][idx_zc]
        
        # If phi has no zero crossing, set roll frequency to None
        if t_start.size == 0:
            f_avg, f_std = None, None
        # Else, approxomate roll frequency as average time period
        # between zero crossings
        else:                    
            f = 0.5 / np.diff(t_start)                        
            f_avg = np.mean(f)
            f_std = np.std(f) 
                          
        return f_avg, f_std, phi
        