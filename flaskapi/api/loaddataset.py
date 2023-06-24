import h5py
import numpy as np
from flask import current_app

def load_dataset(hdf):
    """
    hdfからデータを読み込み、時系列順に並べなおしたndarrayを返す
    
    Parameters
    ----------
    hdf : hdfファイルのパス

    Returns
    -------
    dataValue : hdf5からデータを読み込み、時系列順に並べなおしたndarray
    """

    with h5py.File(hdf, mode='r') as dataset: 
        file_basename = current_app.config["BASEHDF"]
        label= current_app.config["BASENPY"]
        shape = dataset[f'{file_basename}_files'].shape
        # print(f'dataset[{file_basename}_files].shape:{shape}')   
        recording_ids = [i.decode('utf-8') for i in dataset[f'{file_basename}_labels']]
        data = {k:v for k, v in zip(recording_ids, dataset[f'{file_basename}_files'])}
        # print(data)
        dataValue = np.zeros(shape, dtype=np.float32)
        # print('dataValue.shape:',dataValue.shape)
        for i in range(len(data)):
            keyname = str(i)+'_'+label
            dataValue[i] = data[keyname]

    return dataValue