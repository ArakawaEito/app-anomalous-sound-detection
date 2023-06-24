from pathlib import Path

class Config:
    TESTING = False
    DEBUG = False

    SECRET_KEY="897-4r83yhfh-e8rg-ewfo@a"

    basedir=Path(__file__).parent.parent.parent
    # tmpファイルの保存先
    TMPDIR=str( basedir / "tmp_data")

    # 学習済みモデルのパス
    MODELPATH=str( basedir / "data" / "trainedModel"/ "AutoEncoder.hd5")

    BASEHDF="tmp" # hdfファイルのデータセットにつけるベースとなる名前
    BASENPY="newNpy" # npyに変換する際のベースとなる名前
    
    # 音関係のパラメータ
    SR=16000 # サンプリング周波数
    DURATION=1 # メルスペクトログラムの時間幅[s]
    MELPARAMS={'sr':SR, 'n_mels':128, 'fmin':0, 'fmax':SR/2} # メルスペクトログラムのパラメータ

    # 埋もれているか埋もれていないかの閾値
    XDIFF=0.27