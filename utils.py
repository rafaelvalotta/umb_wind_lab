#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Collection of functions for training surrogate models and plotting their
output and error.
"""

# %% Import.

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import get_cmap
# import mplcursors
import base64
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf


# %% Helper functions.

def set_tf_seed(os_seed=0, random_seed=1, np_seed=2, tf_seed=3, inter_seed = 1, intra_seed= 1):
    """
    Set several seeds used by TensorFlow, to ensure the reproducibility of the results.
    """
    os.environ['PYTHONHASHSEED'] = str(os_seed)
    random.seed(random_seed)
    np.random.seed(np_seed)
    tf.random.set_seed(tf_seed)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(inter_seed)
        tf.config.threading.set_intra_op_parallelism_threads(intra_seed)
    except RuntimeError:
        # It has probably already been set.
        pass


class NumpyEncoder(json.JSONEncoder):
    """
    Allows to write numpy arrays in json files.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def save_extra_data(
        folder,
        input_scalers,
        output_scalers,
        data_path,
        extra_data,
        input_transformers=None):
    """
    Save extra data for evaluation of models created by TensorFlow.

    The data will be saved in a folder. This folder will contain 
    a json file with extra data useful to evaluate the surrogate.

    Parameters
    ----------
    folder : str
        Folder that will contain the surrogate model.
    input_scalers : dict of MinMaxScaler
        All input scalers.
    input_transformers : dict of PowerTransformer
        All input transformers.
    output_scalers : dict of MinMaxScaler
        All output scalers
    extra_data : dict
        All extra data needed to correctly evaluate the surrogate.
    """
    # Create the folder.
    os.makedirs(folder, exist_ok=True)

    # Add the input scalers to the extra data.
    extra_data['input_scalers'] = {}
    for set_name, input_scaler in input_scalers.items():
        extra_data['input_scalers'][set_name] = {}  # This avoids to store the __dict__ in a tuple.
        extra_data['input_scalers'][set_name] = input_scaler.__dict__

    # Add the input transformers to the extra data.
    if input_transformers is not None:
        extra_data['input_transformers'] = {}
        for set_name, input_transformer in input_transformers.items():
            extra_data['input_transformers'][set_name] = {}  # This avoids to store the __dict__ in a tuple.
            # The PowerTransformer contains some easy attributes, and a StandardScalar.
            # Let us loop over them.
            for attr_name, attr in input_transformer.__dict__.items():
                if type(attr) is not StandardScaler:
                    # Just add it.
                    extra_data['input_transformers'][set_name][attr_name] = attr
                else:  # type(attr) is StandardScaler
                    # attr_name should be _scaler
                    # We need to convert this object into a dict.
                    extra_data['input_transformers'][set_name][attr_name] = attr.__dict__
    # Add data path 
    extra_data['data_path'] = data_path
    # Add the output scalers to the extra data.
    extra_data['output_scalers'] = {}
    for set_name, output_scaler in output_scalers.items():
        extra_data['output_scalers'][set_name] = {}  # This avoids to store the __dict__ in a tuple.
        extra_data['output_scalers'][set_name] = output_scaler.__dict__

    # Save the extra data.
    with open(f'{folder}/extra_data.json', 'w', encoding='utf-8') as fid:
        json.dump(extra_data, fid, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


def load(*args, **kwargs):
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.load(*args, **kwargs)

def load_extra_data(model_folder):
    with open(model_folder, 'r', encoding='utf-8') as fid:
        extra_data = load(fid)
    
    surrogate = {}
    # Loop over the sets.
    for set_name in extra_data['input_scalers'].keys():
        surrogate[set_name] = {}
        # Create the input and output scaler objects.
        surrogate[set_name]['input_scaler'] = MinMaxScaler()
        surrogate[set_name]['output_scaler'] = MinMaxScaler()
        # Set their attributes.
        for name in extra_data['input_scalers'][set_name].keys():
            if type(extra_data['input_scalers'][set_name][name]) is list:
                setattr(surrogate[set_name]['input_scaler'],
                        name,
                        np.array(extra_data['input_scalers'][set_name][name]))
                setattr(surrogate[set_name]['output_scaler'],
                        name,
                        np.array(extra_data['output_scalers'][set_name][name]))
            else:
                setattr(surrogate[set_name]['input_scaler'],
                        name,
                        extra_data['input_scalers'][set_name][name])
                setattr(surrogate[set_name]['output_scaler'],
                        name,
                        extra_data['output_scalers'][set_name][name])
    if "input_channel_names" in extra_data:
        surrogate['input_channel_names'] = extra_data['input_channel_names']
    if "output_channel_name" in extra_data:
        surrogate['output_channel_names'] = extra_data['output_channel_name']    
    if "testing_index" in extra_data:
        surrogate['testing_index'] = extra_data['testing_index']      
    if "data_path" in extra_data:
        surrogate['data_path'] = extra_data['data_path']      
    if "coords" in extra_data:
        surrogate['coords'] = extra_data['coords']        
    return surrogate
    
def cartesian_product(*arrays):
    """
    Compute the cartesian product of the given arrays.

    This function has been copied from https://stackoverflow.com/a/11146645/3676517
    and https://stackoverflow.com/a/45378609/3676517

    Parameters
    ----------
    arrays : array_like
        Input arrays.

    Returns
    -------
    array_like
        Cartesian product of the input arrays.

    """
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

# def xr_coords_cartesian_product(xr_coords):
#     """
#     Compute the cartesian product of xarray dataset coords.

#     Parameters
#     ----------
#     xr_coords : xarray coords.

#     Returns
#     -------
#     array_like
#         Cartesian product of the input arrays.

#     """
#     arr_dims = list(xr_coords.keys())
#     arrays = []
#     for dim in arr_dims:
#         arrays.append(xr_coords[dim].values.astype(np.float32))

#     la = len(arrays)
#     dtype = np.find_common_type([a.dtype for a in arrays], [])
#     arr = np.empty([np.size(a) for a in arrays] + [la], dtype=dtype)
#     for i, a in enumerate(np.ix_(*arrays)):
#         arr[..., i] = a
#     return arr.reshape(-1, la)

# %% Classes.

class EarlyStoppingByLossVal(tf.keras.callbacks.Callback):
    """
    Callback to stop training when the monitored quantity does not improve after a given value.
    """
    # Inspired from: https://stackoverflow.com/a/37296168/3676517

    def __init__(self, monitor='loss', value=1e-3, mode='min'):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        if monitor not in ('loss', 'val_loss'):
            raise ValueError('monitor must be "loss" or "val_loss".')
        self.value = value
        self.mode = mode
        if mode not in ('min', 'max'):
            raise ValueError('mode must be "min" or "max".')

    def on_epoch_end(self, epoch, logs={}):
        current = logs[self.monitor]
        if (self.mode == 'min') and (current < self.value):
            self.model.stop_training = True
        elif (self.mode == 'max') and (current > self.value):
            self.model.stop_training = True

