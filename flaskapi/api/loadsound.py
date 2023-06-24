import librosa as lb
from flask import current_app

import os

ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_sound(request, session):
    # requestごとに固有のフォルダを作る。同時アクセスがあったときに別々のファイルにアクセスするようにするため
    dir_session = session["uploadID"]
    dir_tmpSound =  current_app.config["TMPDIR"]+"/wav/"+dir_session
    os.makedirs(dir_tmpSound, exist_ok=True)

    file = request.files['soundFile']
    if file and allowed_file(file.filename):
        # requestのwavファイルをtmp.wavとして保存
        savePath = os.path.join(dir_tmpSound, "tmp.wav")
        file.save(savePath)
        y, sr = lb.load(savePath)

    return y, sr