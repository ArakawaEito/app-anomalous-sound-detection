from .preprocess import convert_sound
from .loaddataset import load_dataset
from .postprocess import delete_dirs

from flask import current_app, jsonify

import numpy as np
from tensorflow.keras import models
import tensorflow as tf

def calculation(request, session):
    """
    異常度の算出

    Parameters
    ----------
    hdf : hdfファイルのパス

    Returns
    -------
    buried_sound : hdf5からデータを読み込み、時系列順に並べなおしたndarray

    """
    xdiff = current_app.config["XDIFF"] # 埋もれているか埋もれていないかの閾値

    # モデルのロード
    modelPath = current_app.config["MODELPATH"]
    AutoEncoder = models.load_model(modelPath)
    print(AutoEncoder.summary())

    # アップロードされたwav形式ファイルをhdf形式に変換しストレージに保存
    convert_sound(request, session)

    # hdfからデータを読み込み、時系列順に並べなおしたndarrayを返す
    dir_session = session["uploadID"]
    file_newhdf = current_app.config["TMPDIR"]+"/hdf5/"+dir_session+"/newhdf.hdf5"
    print(file_newhdf)
    calcDataValue = load_dataset(file_newhdf) # type : ndarray, 異常度計算対象のデータ

    # 異常度の計算
    predict_batch_size=256
    decoded_imgs = np.empty(calcDataValue.shape, dtype=np.float32)  
    buried_sound = np.zeros(len(calcDataValue)) # 異常音が埋もれているかどうかの判断に使用

    BATCH_INDICES = np.arange(start=0, stop=len(calcDataValue), step=predict_batch_size)  
    BATCH_INDICES = np.append(BATCH_INDICES, len(calcDataValue)) 

    for index in (np.arange(len(BATCH_INDICES) - 1)):
        batch_start = BATCH_INDICES[index]  
        batch_end = BATCH_INDICES[index + 1] 
        decoded_imgs[batch_start:batch_end] = AutoEncoder.predict_on_batch(calcDataValue[batch_start:batch_end]) 
        
        # 差分画像の最大値と復元画像の平均値の比を使ってBG音に埋もれているかどうかを判断.
        diff = (calcDataValue[batch_start:batch_end]-decoded_imgs[batch_start:batch_end])     
        diff_max = diff.mean(axis=3).max(axis=1).max(axis=1) 
        decoded_imgs_mean = decoded_imgs[batch_start:batch_end].mean(axis=3).mean(axis=1).mean(axis=1)        
        buried_sound[batch_start:batch_end] = diff_max/decoded_imgs_mean
        
    anomary_scores = np.zeros(len(decoded_imgs))
    for i in range(len(anomary_scores)):
        anomary_scores[i] = np.mean((decoded_imgs[i]-calcDataValue[i])**2)    
    buried_sound  = np.where(buried_sound>=xdiff, 1, 0) # １なら埋もれていない，0なら埋もれている   
    notburied_anomary_scores = np.where(buried_sound==1, anomary_scores, 0)  
    # print("result:", notburied_anomary_scores.tolist())    

    # 使わなくなったフォルダを削除
    delete_dirs(session)

    return jsonify({'predictions' : notburied_anomary_scores.tolist()})
