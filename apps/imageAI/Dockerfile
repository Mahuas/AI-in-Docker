FROM python:3.6
WORKDIR /home
COPY . /home
RUN apt-get update
RUN pip install --default-timeout=1000 -r ./requirements.txt
EXPOSE 3600
CMD ["python","main.py"]
