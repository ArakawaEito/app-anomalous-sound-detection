FROM python:3.11

WORKDIR usr/src/app

COPY ./requirements.txt .
COPY ./app.py .
COPY ./.streamlit ./.streamlit
RUN pip install -r requirements.txt

EXPOSE 8501
