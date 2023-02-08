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
        output['t'] = []
        
        output['exit_status'] = []
                        
        for filepath in self.sim_filepaths: 
                    
            data = pickle.load(open(filepath, 'rb'))

            for key in FS_keys:
                
                # TODO: Do dot_W_lin and dot_W_rot need to be handled differently?
                output['FS'][key].append(getattr(data['FS'], key))
            
            for key in CS_keys: output['CS'][key].append(getattr(data['CS'], key))
            
            output['t'].append(data['FS'].times)
            output['exit_status'].append(data['exit_status'])
                                
        # Check whether simulation time is a grid parameter
        self.T_is_param = False
        keys = self.PG.keys
        
        if type(keys) == list:
            for key in keys:
                if self.T_is_param: break                                                                
                if type(key) == tuple:                 
                    for k in key:
                        if self.T_is_param: break                                                
                        if 'T' in k: 
                            self.T_is_param = True 
                            break
                else: 
                    if key == 'T':
                        self.T_is_param = True
                        break
        else:
            if type(keys) == tuple:
                for k in keys:
                    if 'T' in k:
                        self.T_is_param = True
                        break
            else: 
                if keys == 'T':
                    self.T_is_param = True
                
        # If simulation time is the same for all simulations
        # time array only needs to be stored once                        
        if not self.T_is_param:                            
            #TODO: t could possibly start not at dt
            output['t'] = output['t'][0]
        
        return output

    def pad_arrays(self, arr_list, exit_status_arr, T_list = None):
        '''
        Simulations which are failed need to be padded with nans.
        
        :param arr_list (list): 
        :param exit_status (int):
        :param T_list (list): List with different simulation times        
        '''                        
        #TODO: Fail potential
        if self.PG.base_parameter['dt_report'] is not None:
            dt = self.PG.base_parameter['dt_report']
        else:
            dt = self.PG.base_parameter['dt']

        if not self.T_is_param:            
            n = int(round(self.PG.base_parameter['T']/dt))        
            n_list = len(arr_list)*[n]
        else:
            n_list = [int(round(T/dt)) for T in T_list]
        
        # Desired shape
        shape_list = [(n,) + arr_list[0].shape[1:] for n in n_list]                        
                               
        pad_arr_list = []
                                                        
        for arr, exit_status, shape in zip(arr_list, exit_status_arr, shape_list):
            
            # If simulation succeded, no padding is needed
            if exit_status == 0:
                pad_arr_list.append(arr)
                continue
            
            try:
                # If simulation failed we pad the missing time steps with nans
                pad_arr = np.zeros(shape)
                pad_arr[:] = np.nan                    
                pad_arr[:np.size(arr,0)] = arr
            except Exception as e:
                print(f'Debug: {e}')
                        
            pad_arr_list.append(pad_arr)
            
        return pad_arr_list
            
    def add_data_to_h5(self, h5, FS_keys, CS_keys = []):
        
        if self.PG.filename not in h5:
        
            if type(FS_keys) == str: FS_keys = [FS_keys]
            if type(CS_keys) == str: CS_keys = [CS_keys]
            
            output = self.load_data(FS_keys, CS_keys)
                    
            PG_grp = h5.create_group(self.PG.filename)                
            FS_grp = PG_grp.create_group('FS')
            CS_grp = PG_grp.create_group('CS')
        
            if self.T_is_param:
                T_list = [param['T'] for param in self.PG.param_arr]
                
            exit_status = output['exit_status']
                                    
            for key, arr in output['FS'].items():            

                if not self.T_is_param:        
                    arr = self.pad_arrays(arr, exit_status)            
                    FS_grp.create_dataset(key, data = arr)
                else:
                    arr = self.pad_arrays(arr, exit_status, T_list)                            
                    grp = FS_grp.create_group(key)                                
                    for (a,_hash) in zip(arr, self.PG.hash_arr):
                        grp.create_dataset(_hash, data = a)
    
            for key, arr in output['CS'].items():            
                
                if not self.T_is_param:                    
                    CS_grp.create_dataset(key, data = arr)
                else:
                    grp = CS_grp.create_group(key)
                    for (a,_hash) in zip(arr, self.PG.hash_arr):
                        grp.create_dataset(_hash, data = a)

            PG_grp.create_dataset('t', data = output['t'])
            PG_grp.create_dataset('exit_status', data = exit_status)                
            PG_grp.attrs['grid_filename'] = self.PG.filename + '.json'       
                
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

        if self.T_is_param:
            T_list = [param['T'] for param in self.PG.param_arr]
        
        for key, arr in output['FS'].items():            
        
            if not self.T_is_param:        
                arr = self.pad_arrays(arr, exit_status)            
                FS_grp.create_dataset(key, data = arr)
            else:
                arr = self.pad_arrays(arr, exit_status, T_list)                            
                grp = FS_grp.create_group(key)                                
                for (a,_hash) in zip(arr, self.PG.hash_arr):
                    grp.create_dataset(_hash, data = a)

        CS_grp = h5.create_group('CS')

        for key, arr in output['CS'].items():            

            if not self.T_is_param:                    
                CS_grp.create_dataset(key, data = arr)
            else:
                grp = CS_grp.create_group(key)
                for (a,_hash) in zip(arr, self.PG.hash_arr):
                    grp.create_dataset(_hash, data = a)

        if not self.T_is_param:                        
            h5.create_dataset('t', data = output['t'])
        else:
            grp = h5.create_group('t')
            for (t,_hash) in zip(output['t'], self.PG.hash_arr):
                grp.create_dataset(_hash, data = t)
                        
        h5.create_dataset('exit_status', data = exit_status) 
        
        h5.attrs['grid_filename'] = self.PG.filename + '.json'       
                                        
        return h5
                
                                                                                                           

