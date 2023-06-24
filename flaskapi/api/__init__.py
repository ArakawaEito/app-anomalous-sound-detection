from flask import Blueprint, jsonify, request, session
import datetime

from flaskapi.api import calcAnomalyScore
api = Blueprint("api", __name__)


@api.get("/")
def index():
    return jsonify({"column": "value"}), 201

@api.post("/sound")
def loadsound():
    id = datetime.datetime.now()
    session["uploadID"] = id.strftime("%Y%m%d%H%M%S%f")
    print("uploadID:",session["uploadID"])

    return calcAnomalyScore.calculation(request, session)
