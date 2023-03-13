'''
Created on 17 Sept 2022

@author: lukas
'''
from os.path import join, isfile
import pickle
import h5py
import numpy as np
import warnings

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
        Loads, pools and saves simulation results for given grid parameter file
        to HDF5.
        
        :param grid_param_path (str): grid_param file path  
        :param sim_path (str): Simulation file directory 
        '''
                
        self.PG = ParameterGrid.init_pg_from_filepath(grid_param_path)                
        self.sim_path = sim_path
                  
        return
                                                                                  
    @property                        
    def sim_filepaths(self):

        return [join(self.sim_path, h + '.dat') for h in self.PG.hash_arr]
        
    def load_data(self, 
                  FS_keys, 
                  CS_keys = []):                            
        '''        
        Loads results from pickled FrameSequence and ControlSequence
        specified by given keys. 
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
                
                output['FS'][key].append(getattr(data['FS'], key))
            
            for key in CS_keys: output['CS'][key].append(getattr(data['CS'], key))
            
            output['t'].append(data['FS'].times)
            output['exit_status'].append(data['exit_status'])
                                                        
        # If the simulation time T is identical for all simulations 
        # time stamps only need to be stored once                        
        if not self.PG.has_key('T'):                            
            #TODO: t could possibly start not at dt
            output['t'] = output['t'][0]
        
        return output

    def pad_arrays(self, 
            arr_list, 
            exit_status_arr):
        '''
        Pads failed simulation results with nans.        
        '''       
        if np.all(np.logical_not(exit_status_arr)):
            return arr_list
                
        pad_arr_list = []
        
        for arr, exit_status, P in zip(arr_list, exit_status_arr, PG.param_arr):
            
            if not exit_status:
                continue
            
            if P['dt_report'] is not None: dt = P['dt_report']
            else: dt = P['dt']
                    
            n = int(round(P['T']/dt))                
            shape = (n,) + arr.shape[1:]                         
            
            pad_arr = np.zeros(shape)[:] = np.nan                                
            pad_arr[:arr.size(0)] = arr
            pad_arr_list.append(pad_arr)
            
        return pad_arr_list
            
    def add_data_to_h5(self, h5, FS_keys, CS_keys = []):
        
        if self.PG.filename in h5:
            print(f'Group for grid {self.PG.filename} already exists')
            return
        
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

            if not self.PG.has_key('T'):        
                arr = self.pad_arrays(arr, exit_status)            
                FS_grp.create_dataset(key, data = arr)
            else:
                arr = self.pad_arrays(arr, exit_status, T_list)                            
                grp = FS_grp.create_group(key)                                
                for (h, a) in zip(self.PG.hash_arr, arr):
                    grp.create_dataset(h, data = a)

        for key, arr in output['CS'].items():            
            
            if not self.PG.has_key('T'):                    
                CS_grp.create_dataset(key, data = arr)
            else:
                grp = CS_grp.create_group(key)
                for (h, a) in zip(self.PG.hash_arr, arr):
                    grp.create_dataset(h, data = a)

        PG_grp.create_dataset('t', data = output['t'])
        PG_grp.create_dataset('exit_status', data = exit_status)                
        PG_grp.attrs['grid_filename'] = self.PG.filename + '.json'       
            
                                
        return

    def save_data(self,
        filepath, 
        FS_keys, 
        CS_keys = [], 
        overwrite = True,
        h5 = None):
        '''
        Chooses which save method to use. If simulation time T
        viaries over the grid then the simulation results 
        can not be pooled into a single dataset because they 
        different shapes in which case we use save_data_pool_T 
        or save_data_no_pool.        
        '''
                
        has_T, dim = self.PG.has_key('T', True)

        if not has_T:
            h5 = self.save_data_pool(filepath,
                FS_keys, CS_keys, overwrite, h5)
        else:
            if dim == 0 and not self.PG.line:
                h5 = self.save_data_pool_T(filepath, 
                    FS_keys, CS_keys, overwrite, h5)
            else:
                h5 = self.save_data_no_pool(filepath, 
                    FS_keys, CS_keys, overwrite, h5)
        
        return h5
                             
    def save_data_pool(self, 
            filepath, 
            FS_keys, 
            CS_keys = [], 
            overwrite = True,
            h5 = None):
        '''
        Pools and saves specified simulation results into single HDF5 file.
        The simulation time is assumed to be identical for all simulations, 
        i.e. results for each key can be pooled into single dataset.
        '''
        assert not self.PG.has_key('T'), ('ParameterGrid has T as a key, simulations times ' 
            'must be identical for all simulations')
        
        if isfile(filepath):
            if not overwrite:
                print(f'HDF5 file {filepath} already exists.' 
                    'Set overwrite=True to overwrite existing file.')
                return
                                                
        if type(FS_keys) == str: FS_keys = [FS_keys]
        if type(CS_keys) == str: CS_keys = [CS_keys]
                                            
        output = self.load_data(FS_keys, CS_keys)
        
        # If not HDF5 file is passed create one
        if h5 is None:
            h5 = h5py.File(filepath, 'w')
            h5.attrs['grid_filename'] = self.PG.filename + '.json'       
        # If a HDF5 file is passed create group
        else:
            if self.PG.filename in h5:
                if not overwrite:
                    print(f'Group {self.PG.filename} in given HDF5 File already exists')
                    return
            h5 = h5.create_group(self.PG.filename)
            
        exit_status_arr = output['exit_status']
        h5.create_dataset('exit_status', data = exit_status_arr) 
                
        FS_grp = h5.create_group('FS')
                
        for key, arr_list in output['FS'].items():            
        
            arr_list = self.pad_arrays(arr_list, exit_status_arr)            
            FS_grp.create_dataset(key, data = np.array(arr_list))
            
        CS_grp = h5.create_group('CS')
            
        for key, arr in output['CS'].items():            
            
            CS_grp.create_dataset(key, data = np.array(arr))
 
        h5.create_dataset('t', data = output['t'])
        
        return
    
    def save_data_no_pool(self,
        filepath, 
        FS_keys, 
        CS_keys = [], 
        overwrite = True,
        h5 = None):
        '''
        Saves each simulation results into a single HDF5 file.
        No pooling, each FrameSequence result is saved as a
        single dataset. Use if simulation times vary over 
        the grid.
        '''
        
        if not PG.has_key('T'):
            warnings.warn('ParameterGrid does not have T as a sweep key, ' 
                'i.e. the simulation T is identical for all simulations,'
                'use save_data instead!')
        
        if isfile(filepath):
            if not overwrite:
                print(f'HDF5 file {filepath} already exists. Set overwrite=True to overwrite existing file.')
                return
                                                
        if type(FS_keys) == str: FS_keys = [FS_keys]
        if type(CS_keys) == str: CS_keys = [CS_keys]

        # If not HDF5 file is passed create one
        if h5 is None:
            h5 = h5py.File(filepath, 'w')
            h5.attrs['grid_filename'] = self.PG.filename + '.json'       
        # If a HDF5 file is passed create group
        else:
            if self.PG.filename in h5:
                if not overwrite:
                    print(f'Group {self.PG.filename} in given HDF5 File already exists')
                    return
            h5 = h5.create_group(self.PG.filename)
                                            
        output = self.load_data(FS_keys, CS_keys)
        
        h5.attrs['grid_filename'] = self.PG.filename + '.json'       
        exit_status_arr = output['exit_status']
        h5.create_dataset('exit_status', data = exit_status_arr) 
        
        for k in ['FS', 'CS']:

            grp = h5.create_group(k)        
        
            for key, arr_list in output[k].items():            

                key_grp = grp.create_group(key)
                
                # Simulations can fail in which case we pad the results
                if k == 'FS':
                    arr_list = self.pad_arrays(arr_list, exit_status_arr)            
                                                                 
                for h, arr in zip(self.PG.hash_arr, arr_list):
                    key_grp.create_dataset(h, data = arr)

                for h, arr in zip(self.PG.hash_arr, arr_list):
                    key_grp.create_dataset(h, data = arr)

        for h, t in zip(self.PG.hash_arr, output['t']):
                grp.create_dataset(h, data = t)
                                                                        
        return h5    
    
    
    def save_data_pool_T(self, 
        filepath, 
        FS_keys, 
        CS_keys = [], 
        overwrite = True,
        h5 = None):
        '''
        Pools and saves specified simulation results into single HDF5 file.
        The simulation time is assumed to vary over the first dimension 
        of the given parameter grid. Different results are therefore saved 
        as N datasets where N is the size of the ParameterGrid's first 
        dimension each having the shape of the remaining dimensions.         
        '''
        
        has_T, dim = self.PG.has_key('T', return_dim=True)
        assert has_T, 'ParameterGrid does not has T as a key'
        assert dim==0, "T must be varied over the ParameterGrid's first dimension"
        
        if isfile(filepath):
            if not overwrite:
                print(f'HDF5 file {filepath} already exists.' 
                    'Set overwrite=True to overwrite existing file.')
                return
                
        if type(FS_keys) == str: FS_keys = [FS_keys]
        if type(CS_keys) == str: CS_keys = [CS_keys]
        
        
        # If not HDF5 file is passed create one
        if h5 is None:
            h5 = h5py.File(filepath, 'w')
            h5.attrs['grid_filename'] = self.PG.filename + '.json'       
        # If a HDF5 file is passed create group
        else:
            if self.PG.filename in h5:
                if not overwrite:
                    print(f'Group {self.PG.filename} in given HDF5 File already exists')
                    return
            h5 = h5.create_group(self.PG.filename)
        
        # Load data from pickled FrameSequences                                    
        output = self.load_data(FS_keys, CS_keys)
        
        exit_status_arr = output['exit_status']
        h5.create_dataset('exit_status', data = exit_status_arr) 

        T_arr = self.PG.v_from_key('T')
        h5.create_dataset('T', data = T_arr)
        
        t_list = []
        
        for k in ['FS', 'CS']:
        
            grp = h5.create_group(k)        
        
            for key, arr_list in output[k].items():            
        
                key_grp = grp.create_group(key)
                
                # Simulations can fail in which case we pad the results
                if k == 'FS':
                    arr_list = self.pad_arrays(arr_list, exit_status_arr)            
                    
                # Pool results with the same simulation time T
                for i, T in enumerate(T_arr):                                                        
                    sub_arr_list = []                    
                    
                    for idx in self.PG.flat_index(self.PG[i, :]):
                        sub_arr_list.append(arr_list[idx])                                                            
                    
                    key_grp.create_dataset(f'{T}', data = np.array(sub_arr_list))
                    # Time stamps are identical for results associated with 
                    # the same simulation time T, i.e. we only need to save them 
                    # once for each iteration of the outer for loop
                    t_list.append(output['t'][idx])
                                                
        grp = h5.create_group('t')                 
        for t, T in zip(t_list, T_arr): grp.create_dataset(f'{T}', data = t)
                                                                            
        return h5        
            
