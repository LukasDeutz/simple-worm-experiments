'''
Created on 10 Feb 2023

@author: lukas

Default model parameter  
'''

def default_model_parameter():
    
    parameter = {}
    
    # Simulation parameter          
    parameter['N']  = 100
    parameter['dt'] = 0.01
    parameter['N_report'] = 100
    parameter['dt_report'] = 0.01

    # Solver parameter
    parameter['pi'] = False
    parameter['pi_alpha0'] = 0.9            
    parameter['pi_maxiter'] = 1000             
    parameter['fdo'] = {1: 2, 2:2}
      
    # Model parameter    
    parameter['use_inertia'] = False
    
    # Fluid parameter
    parameter['external_force'] = ['rft']    
    parameter['rft'] = 'Whang'
    parameter['mu'] = 1e-3
                
    # Geometric parameter    
    parameter['L0'] = 1130 * 1e-6
    parameter['r_max'] = 32 * 1e-6
    parameter['rc'] = 'spheroid'
    parameter['eps_phi'] = 1e-3
            
    # Material parameter    
    parameter['E'] = 1e5  
    parameter['G'] = parameter['E'] / (2 * (1 + 0.5))
    parameter['eta'] = 1e-2 * parameter['E']
    parameter['nu'] = 1e-2 * parameter['G']

    # Muscle parameter            
    parameter['mts'] = True
    parameter['tau_on'] = 0.05
    parameter['Dt_on'] = 3*parameter['tau_on']
        
    return parameter
