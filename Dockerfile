FROM python:3.9

ADD tenzor.py .

RUN pip install tensorflow==2.9.1 scikit-learn

CMD ["python", "tenzor.py"]