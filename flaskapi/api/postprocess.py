import shutil
import os

from flask import current_app

def delete_dirs(session):
    """アップロードされたwavファイルと作成したhdfを削除"""

    # wavファイルとhdfが保存されているフォルダのパスを指定して削除
    dir_session = session["uploadID"]
    dir_tmpSound = current_app.config["TMPDIR"]+"/wav/"+dir_session

    dir_newhdf = current_app.config["TMPDIR"]+"/hdf5/"+dir_session

    if(os.path.isdir(dir_tmpSound) == True):
        shutil.rmtree(dir_tmpSound)
        print(f'{dir_tmpSound}削除完了')

    if(os.path.isdir(dir_newhdf) == True):
        shutil.rmtree(dir_newhdf)
        print(f'{dir_newhdf}削除完了')