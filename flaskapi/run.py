import os
from flask import Flask
from flaskapi.api import api
from flaskapi.api.config import config

config_name = os.environ.get("CONFIG", "local")

app = Flask(__name__)
# print("name:", __name__)
app.config.from_object(config[config_name])
# Blueprintをアプリケーションに登録
app.register_blueprint(api)
# print(app.config)
