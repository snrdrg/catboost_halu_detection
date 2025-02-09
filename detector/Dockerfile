FROM python:3.11-slim
EXPOSE 8088
RUN mkdir /home/decoder
RUN apt-get update
RUN apt-get install python3-pip -y
ADD . /home/decoder
RUN pip install --no-cache-dir -r /home/decoder/requirements.txt
WORKDIR /home/decoder
RUN chmod +x /home/decoder/db_init.py
RUN python3 /home/decoder/db_init.py
RUN pip install uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]