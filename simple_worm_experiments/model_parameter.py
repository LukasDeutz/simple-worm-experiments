'''        
Created on 10 Feb 2023

@author: lukas

Default model parameter  
'''
from argparse import ArgumentParser
    
def default_model_parameter(as_dict = True):
    
    param = ArgumentParser(description = 'model-parameter')
        
    # Simulation parameter         
    param.add_argument('--N', type = int, default = 100, 
        help = 'Number of centreline points')
    param.add_argument('--dt', type = float, default = 0.01, 
        help = 'Time step')
    param.add_argument('--N_report', type = int, default = 100, 
        help = 'Simulation results get saved for N_report centreline points')
    param.add_argument('--dt_report', type = float, default = 0.01, 
        help = 'Simulation results get saved every dt_report time step')

    # Solver parameter
    param.add_argument('--pi', type = bool, default = False, 
        help = 'If true, solver does a Picard iteration at every time step')
    param.add_argument('--pi_alpha0', type = float, default = 0.9, 
        help = 'Learning rate, employed by the Picard iteration')
    param.add_argument('--pi_maxiter', type = int, default = 1000, 
        help = 'Maximum number of Picard iterations')
    param.add_argument('--fdo', nargs = 2, type = int, default = [2, 2], 
        help = 'Order of finite difference approximations')
    param.add_argument('use_inertia', type = bool, default = False, 
        help = 'If true, interia terms are included into the equations of motion')
              
    # Fluid parameter
    param.add_argument('--fsi', type = str, default = 'rft',
        choices = ['rft', 'sbt'],
        help = 'Model used to approximate the fluid body surface interaction')
    param.add_argument('--rft', type = str, default = 'Whang',
        choices = ['Whang'], #TODO: Add other options
        help = 'Name of resistive-force theory to be used, only relevant if fsi is set to rft')
    param.add_argument('--mu', type = float, default = 1e-3, 
        help = 'Fluid viscosity')
                
    # Geometric parameter    
    param.add_argument('--L0', type = float, default = 1130 * 1e-6, 
        help = 'Natural worm length')
    param.add_argument('--r_max', type = float, default = 32 * 1e-6, 
        help = 'Maximum worm radius')
    param.add_argument('--rc', type = str, default = 'spheroid', 
        help = 'Natural worm shape')
    param.add_argument('--eps_phi', type = float, default = 1e-3, 
        help = 'Radius at the tip of the head and tale')
                
    # Material parameter
    param.add_argument('--E', type = float, default = 1e5, 
        help = "Young's modulus")
    param.add_argument('--G', type = float, default = 1e5 / (2 * (1 + 0.5)), 
        help = 'Shear modulus')
    param.add_argument('--eta', type = float, default = 1e-2 * 1e5, 
        help = 'Extensional viscosity')
    param.add_argument('--nu', type = float, default = 1e-2 * 1e5 / (2 * (1 + 0.5)), 
        help = 'Shear viscosity')

    # Muscle parameter 
    param.add_argument('--gmo', type = bool, default = True,
        help = 'If true, muscles have a gradual onset at the head and tale')
    param.add_argument('--Ds_h', type = bool, default = 0.01,
        help = 'Sigmoid slope at the head')
    param.add_argument('--Ds_t', type = bool, default = 0.01,
        help = 'Sigmoid slope at the tale')    
    param.add_argument('--s0_h', type = bool, default = 3*0.01,
        help = 'Sigmoid midpoint at the head')
    param.add_argument('--s0_t', type = bool, default = 1-3*0.01,
        help = 'Sigmoid midpoint at the tale')
        
    # Make backwards compatible
    if as_dict: 
        args = param.parse_args()
        param = vars(args) 
                
    return param
