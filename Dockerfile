FROM python:3.8-slim-buster
COPY . /app
WORKDIR /app
RUN apt update -y && apt install awscli -y
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python3", "application.py" ]