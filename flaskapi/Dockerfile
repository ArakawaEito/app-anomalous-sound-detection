FROM python:3.11

WORKDIR /usr/src/

RUN apt-get update

COPY ./api /usr/src/flaskapi/api
COPY ./data /usr/src/flaskapi/data
COPY ./tmp_data /usr/src/flaskapi/tmp_data
COPY ./__init__.py /usr/src/flaskapi/__init__.py
COPY ./requirements.txt /usr/src/flaskapi/requirements.txt
COPY ./run.py /usr/src/flaskapi/run.py

RUN pip install --upgrade pip

RUN pip install -r ./flaskapi/requirements.txt

RUN echo "building..."

ENV FLASK_APP "run.py"

EXPOSE 8000

# CMD ["gunicorn", "flaskapi.run:app"]
CMD ["gunicorn", "-b", "0.0.0.0:8000", "flaskapi.run:app", "-t", "300"]