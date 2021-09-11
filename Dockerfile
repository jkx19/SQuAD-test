# syntax=docker/dockerfile:1


FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

WORKDIR /app

RUN pip3 install transformers==4.8.2\
    datasets==1.9.0

# COPY . .

CMD ["python3"]

# # cpu
# FROM python:3.8-slim-buster

# WORKDIR /app

# RUN pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install transformers==4.8.2\
#     datasets==1.9.0

# COPY . .

# CMD [ "python3", "src/run.py"]