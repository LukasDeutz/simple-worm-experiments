'''
Created on 17 Sept 2022

@author: lukas
'''

# This needs move to go simple-worm-experiments
from os.path import join
import pickle
import h5py

from parameter_scan.util import load_grid_param
from parameter_scan import ParameterGrid

class GridPoolLoader():
                    
    def __init__(self, grid_param_path_list, sim_path):
                        
        
        print(grid_param_path_list)
        self.GridLoaders = [GridLoader(gpp, sim_path) for gpp in grid_param_path_list]                

    def __len__(self):
        
        return len(self.GridLoaders)

    def v_arr(self, idx, key = None):
        
        return self.GridLoaders[idx].v_arr(key)
              
    def save_data(self, file_path, FS_keys, CS_keys = []):
        
        h5 = h5py.File(file_path, 'w')
                                
        for GL in self.GridLoaders:
            
            print(GL.PG.filename)
            GL.add_data_to_h5(h5, FS_keys, CS_keys)
                  
        h5.close()
                                                                                                                   
class GridLoader():

    def __init__(self, 
                 grid_param_path, 
                 sim_path):
        
        self.PG = self._init_PG(grid_param_path)
        self.sim_path = sim_path
                  
    def _init_PG(self, grid_param_path):
        
        print(f'grid_path: {grid_param_path}')
        grid_param, base_parameter = load_grid_param(grid_param_path)           
        PG = ParameterGrid(base_parameter, grid_param)
           
        return PG
    
    def v_arr(self, idx, key = None):
        
        if key is None: 
            return self.PG.v_arr        
        else:
            idx = self.PG.keys.index(key)
            return self.PG.v_arr[idx]
                                                                              
    @property                        
    def sim_filepaths(self):

        return [join(self.sim_path, _hash + '.dat') for _hash in self.PG.hash_arr]
        
    def load_data(self, 
                  FS_keys, 
                  CS_keys = []):                            
        '''        
        Loads FrameSequenceNumpy and returns given keys 
        
        :param keys (list[str], str): keys to load
        '''
        
        if type(FS_keys) == str: FS_keys = [CS_keys]
        if type(CS_keys) == str: CS_keys = [CS_keys]
          
        output = {}                                
        output['FS'] = {key: [] for key in FS_keys}        
        output['CS'] = {key: [] for key in CS_keys}                                                            
        
        output['exit_status'] = []
        
        for filepath in self.sim_filepaths: 
        
            data = open(filepath, 'rb')
            
            data = pickle.load(data)

            for key in FS_keys: output['FS'][key].append(getattr(data['FS'], key))
            for key in CS_keys: output['CS'][key].append(getattr(data['CS'], key))
            
            output['exit_status'].append(data['exit_status'])
        
        #TODO: base_parameter                                        
        #output['parameter'] = self.PG.base_parameter
                                            
        return output

    def add_data_to_h5(self, h5, FS_keys, CS_keys = []):
        
        output = self.load_data(FS_keys, CS_keys)

        print(self.PG.filename)

        PG_grp = h5.create_group(self.PG.filename)                
        FS_grp = PG_grp.create_group('FS')
        CS_grp = PG_grp.create_group('CS')

        for key, arr in output['FS'].items():            
            FS_grp.create_dataset(key, data = arr)

        for key, arr in output['CS'].items():            
            CS_grp.create_dataset(key, data = arr)
                        
        #TODO: Save base parameter                
        
        PG_grp.attrs['exits_status'] = output['exit_status']
        
        return
                
    def save_data(self, filepath, FS_keys, CS_keys = []):
        
        output = self.load_data(FS_keys, CS_keys)
        
        h5 = h5py.File(filepath)
        grp = h5.create_group('FS')
        
        for key, arr in output['FS'].items():            
            grp.create_dataset(key, data = arr)

        grp = h5.create_group('CS')

        for key, arr in output['CS'].items():            
            grp.create_dataset(key, data = arr)

        h5.attrs['exit_status'] = output['exit_status']
                
        return
                
                                                                                                           

