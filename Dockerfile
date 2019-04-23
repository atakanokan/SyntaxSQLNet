FROM python:3.6

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/requirements.txt

RUN echo "Downloading requirements"
RUN pip install -r requirements.txt
RUN pip install gunicorn

RUN echo "Copying all files"
COPY . /code/

EXPOSE 5000
# ENV FLASK_APP=app/main.py
# CMD ["gunicorn", "-k", "gevent", "-b", ":5000", "app.start_app:app", "-t", "75"]
CMD ["python", "run_flask.py"]