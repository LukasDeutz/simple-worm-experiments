'''
Created on 2 Nov 2022

@author: lukas
'''

# Third party imports
from fenics import Expression, Function
import numpy as np

# Local imports
from simple_worm.controls import ControlsFenics, ControlSequenceFenics
from simple_worm_experiments.experiment import Experiment 
                
#===============================================================================
# Simulate undulation

class ContractionRelaxationExperiment(Experiment):
    '''
    Implements control sequences to model contraction into and relaxation out of 
    fixed shape
    '''

    @staticmethod
    def relaxation_control_sequence(worm, parameter):
        '''
        Initialize controls for contraction relation experiment
        
        :param parameter:
        '''

        # TODO: Think about temporal gradual onset for controls        
        T, k = parameter['T'], parameter['k0']                       
                                                                                                                                    
        sm_on = Experiment.muscle_on_switch(parameter)
        sm_off = Experiment.muscle_off_switch(parameter)        
        sh, st = Experiment.spatial_gmo(parameter)

        Omega_expr = Expression(("sm_on*sm_off*sh*st*k", "0", "0"), 
            degree=1, sm_on = sm_on, sm_off = sm_off, 
            sh = sh, st=st, k = k)   

        sigma_expr = Expression(('0', '0', '0'), degree = 1)    
        sigma = Function(worm.function_spaces['sigma'])
        sigma.assign(sigma_expr)

        CS = []
            
        for t in np.linspace(0, T, int(T/worm.dt)):
            
            sm_on.t = t
            sm_off.t = t
            Omega = Function(worm.function_spaces['Omega'])        
            Omega.assign(Omega_expr)
                                    
            C = ControlsFenics(Omega, sigma)
            CS.append(C)
                        
        CS = ControlSequenceFenics(CS)
        CS.rod = worm
        
        return CS
