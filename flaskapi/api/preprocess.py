from pathlib import Path
import os
import glob
import shutil

from flask import current_app
import numpy as np
import librosa as lb
import h5py

from .loadsound import load_sound

basedir = Path(__file__).parent.parent
# print("basedir", type(basedir))

def convert_sound(request, session):
    """
    アップロードされたwav形式のファイルをhdf形式に変換しストレージに保存する
    
    """
    y, sr = load_sound(request, session)

    duration = current_app.config["DURATION"]
    melparams = current_app.config["MELPARAMS"]

    # 各request固有のフォルダの名前。同時アクセスがあったときに別々のファイルにアクセスするようにするため
    dir_session = session["uploadID"]

    # wavファイルのパスや保存先のパス
    dir_tmpSound = current_app.config["TMPDIR"]+"/wav/"+dir_session
    file_wav = dir_tmpSound+"/tmp.wav"

    dir_newNpy = current_app.config["TMPDIR"]+"/npy/"+dir_session
    # print("dir_newNpy:", dir_newNpy)
    os.makedirs(dir_newNpy, exist_ok=True)

    dir_newhdf = current_app.config["TMPDIR"]+"/hdf5/"+dir_session
    file_newhdf = dir_newhdf+"/newhdf.hdf5"
    # print("dir_newhdf:", dir_newhdf)
    # print("file_newhdf:", file_newhdf) 
    os.makedirs(dir_newhdf, exist_ok=True)    


    numSamples = len(y) # wavデータのサンプル数
    totalTime = numSamples // sr # wavデータの長さ
    numCut = int(totalTime//duration) # 分割後のフレーム数(メルスペクトログラムの数)
    print("numSample:",numSamples)
    print("totalTime:",totalTime)
    print("numCut:",numCut)

    # wavをnpyに変換
    npy(file_wav, dir_newNpy, duration, melparams, numCut)
    # npyを画像データに変換しhdf形式で保存
    hdf(file_newhdf, dir_newNpy)

    # hdf変換後npyファイルは必要ないので削除
    if(os.path.isdir(dir_newNpy) == True):
        print(f'hdfへの変換が完了したため{dir_newNpy}を削除します')
        shutil.rmtree(dir_newNpy)
        print(f'{dir_newNpy}削除完了')

    print("変換成功！！")


def MonoToColor(X, eps=1e-6, 
                mean=-40.240447998046875, 
                std=11.049322128295898):
    """
    3次元画像に変換し正規化

    Parameters
    ----------
    X : 正規化前の一次元画像データ 
    eps: ゼロ除算防止
    mean, std : 正規化用の平均・標準偏差

    Returns
    -------
    V : 正規化した三次元画像データ
    
    """
    mean = mean or X.mean() 
    std = std or X.std()
    Xstd = (X - mean) / (std + eps) # epsでゼロ除算を防ぐ
#     print(type(eps))

    _min, _max = Xstd.min(), Xstd.max()

    if (_max - _min) > eps:
        V = np.clip(Xstd, _min, _max)  
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else: 
        V = np.zeros_like(Xstd, dtype=np.uint8) 

    V = np.stack([V, V, V], axis=-1)
    V = V.astype('float32')/255 
    return V


def npy(file_wav, dir_newNpy, duration, melparams, numCut):
    """
    wavをnpyに変換
    
    Parameters
    ----------
    file_wav : 変換するwavファイルのパス
    dir_newNpy : 変換したnpyの保存先フォルダのパス
    duration : メルスペクトログラムの時間幅[s]
    melparams : メルスペクトログラムのパラメータ
    numCut : 分割後のフレーム数(メルスペクトログラムの数)

    Returns
    -------
    None
    """
    # npy形式で保存
    for i in range(numCut):       
        label= current_app.config["BASENPY"]
        record_name = str(i)+ '_'+label+'.npy'
        y, sr = lb.load(file_wav, sr = None, offset=i*duration, duration=duration)

        melspec = lb.power_to_db(lb.feature.melspectrogram(y=y, **melparams)).astype(np.float32)
        np.save(f'{dir_newNpy}/{record_name}', melspec) # npy形式に保存

def hdf(file_newhdf, dir_newNpy):
    """
    npyを画像データに変換しhdf形式で保存
    
    Parameters
    ----------
    file_newhdf : 変換したhdfファイルの保存先のパス
    dir_newNpy : 変換したnpyの保存先フォルダのパス

    Returns
    -------
    None
    """

    with h5py.File(file_newhdf, mode='w') as f:  
        file_basename=current_app.config["BASEHDF"]
        npyFiles = os.path.join(dir_newNpy, "*.npy")
        # print("npyFiles:", npyFiles)
        npyFiles = glob.glob(npyFiles)
        print('全npyファイル数:', len(npyFiles))  
        
        """ここの処理は時間がかかるので実行しない(精度が落ちる可能性あり)
         if (mean==None)or(std==None):
            mean = []
            std = []
            for i in npyFiles:
                file = np.load(i)
                mean.append(file.mean())# 各データのスペクトログラムの平均値を求め，meanに追加
                std.append(file.std())

            mean = np.array(mean).mean() #各スペクトログラムの平均値をさらに平均してをmeanに格納する
            std = np.array(std).mean()                 
        print(f'mean:{mean}')
        print(f'std:{std}')       
        
        """
        base = np.load(npyFiles[1])
        # print('base:', base.shape)

        baseShape = (len(npyFiles), *base.shape, 3)
        print('baseShape', baseShape) # (データ数，スペクトログラムの行数，スペクトログラムの列数, チャンネル数）

        f.create_dataset(f'{file_basename}_files', baseShape, dtype = np.float32) 
        f.create_dataset(f'{file_basename}_labels', (len(npyFiles),), dtype = 'S50') 

        f[f'{file_basename}_labels'][...] = [os.path.splitext(os.path.basename(i))[0].encode(encoding="ascii", errors = "ignore") for i in npyFiles]
        # f[f'{file_basename}_labels'][...] = [i.split('\\')[-1].split('.')[0].encode(encoding="ascii", errors = "ignore") for i in npyFiles]

        for i, v in enumerate(npyFiles):
            f[f'{file_basename}_files'][i, ...] = MonoToColor(np.load(v))
                
