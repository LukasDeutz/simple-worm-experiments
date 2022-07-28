# Local imports
from simple_worm_experiments.util import default_parameter, dimensionless_MP

if __name__ == '__main__':
    
    parameter = default_parameter()


    E = 1e4
    G = E / (2 * (1 + 0.5))
    eta = 1e-2 * E 
    nu = 1e-2 * G

    parameter['E'] = E
    parameter['G'] = G
    parameter['eta'] = eta
    parameter['nu'] = nu
    parameter['gamma'] = 0.1
    
            
    MP = dimensionless_MP(parameter)    
    print(MP)
    print(MP.c_n / MP.c_t)
    
    MP.to_fenics()
    

    
    
    






