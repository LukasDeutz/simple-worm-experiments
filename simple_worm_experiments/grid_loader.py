'''
Created on 17 Sept 2022

@author: lukas
'''

# This needs move to go simple-worm-experiments
from os.path import join, isfile
import pickle
import h5py
import numpy as np

from parameter_scan.util import load_grid_param
from parameter_scan import ParameterGrid

class GridPoolLoader():
                    
    def __init__(self, grid_param_path_list, sim_path):
                                
        self.GridLoaders = [GridLoader(gpp, sim_path) for gpp in grid_param_path_list]                
        
    def __len__(self):
        
        return len(self.GridLoaders)

    def __getitem__(self, idx):
        
        return self.GridLoaders[idx]

    def __iter__(self):
        
        return iter(self.GridLoaders)
    
    def v_arr(self, idx, key = None):
        
        return self.GridLoaders[idx].v_arr(key)

    @property
    def filenames(self):
        
        return [GL.PG.filename for GL in self]
                      
    def save_data(self, file_path, FS_keys, CS_keys = [], overwrite = True):
        
        if overwrite:
            h5 = h5py.File(file_path, 'w')
        else:
            h5 = h5py.File(file_path, 'r+')
                                
        for GL in self.GridLoaders:
                        
            GL.add_data_to_h5(h5, FS_keys, CS_keys)
                  
        h5.close()
                                                                                                                   
class GridLoader():
    
    def __init__(self, 
                 grid_param_path, 
                 sim_path):
        '''
        
        :param grid_param_path (str): grid_param json filepath
        :param sim_path (str): directory path of simulation result files 
        '''
                
        self.PG = ParameterGrid.init_pg_from_filepath(grid_param_path)                
        self.sim_path = sim_path
                  
    
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
        
        if type(FS_keys) == str: FS_keys = [FS_keys]
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
            
            
        # Time only needs to get stored once        
        T = self.PG.base_parameter['T']
        
        if self.PG.base_parameter['dt_report'] is not None:
            dt = self.PG.base_parameter['dt_report']
        else:
            dt = self.PG.base_parameter['dt']
        
        n = int(T/dt)
        
        #TODO: t could possibly start not at dt
        t = dt * np.arange(1, n+1, 1)       
        output['t'] = t
                
        #TODO: base_parameter                                        
        #output['parameter'] = self.PG.base_parameter
                                            
        return output

    def pad_arrays(self, arr_list, exit_status_arr):
        '''
        Simulations which are failed need to be padded with nans.
        
        :param arr_list (list): 
        :param exit_status (int):
        '''                        
        #TODO: Fail potential
        if self.PG.base_parameter['dt_report'] is not None:
            dt = self.PG.base_parameter['dt_report']
        else:
            dt = self.PG.base_parameter['dt']
            
        n = int(round(self.PG.base_parameter['T']/dt))        
                                    
        # Desired shape
        shape = (n,) + arr_list[0].shape[1:]
                               
        pad_arr_list = []
                                                        
        for arr, exit_status in zip(arr_list, exit_status_arr):
            
            # If simulation succeded, no padding is needed
            if exit_status == 0:
                pad_arr_list.append(arr)
                continue
            
            # If simulation failed we pad the missing time steps with nans
            pad_arr = np.zeros(shape)
            pad_arr[:] = np.nan                    
            pad_arr[:np.size(arr,0)] = arr
            
            pad_arr_list.append(pad_arr)
            
        return pad_arr_list
            
    def add_data_to_h5(self, h5, FS_keys, CS_keys = []):
        
        if self.PG.filename not in h5:
        
            output = self.load_data(FS_keys, CS_keys)
                        
            PG_grp = h5.create_group(self.PG.filename)                
            FS_grp = PG_grp.create_group('FS')
            CS_grp = PG_grp.create_group('CS')
        
            exit_status = output['exit_status']
                                    
            for key, arr in output['FS'].items():            

                arr = self.pad_arrays(arr, exit_status)
                                                                
                FS_grp.create_dataset(key, data = arr)
    
            for key, arr in output['CS'].items():            
                CS_grp.create_dataset(key, data = arr)

            PG_grp.create_dataset('t', data = output['t'])
            PG_grp.create_dataset('exit_status', data = exit_status)        
        
        else:
            print(f'Group for grid {self.PG.filename} already exists')
                                
        return
                
    def save_data(self, filepath, FS_keys, CS_keys = [], overwrite = True):
        
        if isfile(filepath):
            if not overwrite:
                print(f'HDF5 file {filepath} already exists. Set overwrite=True to overwrite existing file.')
                return
                
        if type(FS_keys) == str: FS_keys = [FS_keys]
        if type(CS_keys) == str: CS_keys = [CS_keys]
                                            
        h5 = h5py.File(filepath, 'w')
        
        output = self.load_data(FS_keys, CS_keys)
        
        FS_grp= h5.create_group('FS')
        
        exit_status = output['exit_status']
        
        for key, arr in output['FS'].items():            
        
            arr = self.pad_arrays(arr, exit_status)
            
            FS_grp.create_dataset(key, data = arr)

        CS_grp = h5.create_group('CS')

        for key, arr in output['CS'].items():            
            CS_grp.create_dataset(key, data = arr)

            arr = self.pad_arrays(arr, exit_status)


        h5.create_dataset('t', data = output['t'])
        h5.create_dataset('exit_status', data = exit_status) 
        
        h5.attrs['grid_filename'] = self.PG.filename + '.json'       
                                        
        return h5
                
                                                                                                           

