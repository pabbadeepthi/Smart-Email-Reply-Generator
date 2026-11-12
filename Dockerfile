
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir torch==2.4.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu 

RUN pip install --no-cache-dir \
    torchserve==0.12.0 \
    torch-model-archiver==0.12.0 \
    transformers==4.44.2

COPY handler.py /app/handler.py

RUN mkdir -p /app/model_store

RUN torch-model-archiver \
    --model-name email_reply \
    --version 1.0 \
    --handler /app/handler.py \
    --export-path /app/model_store \
    --force

EXPOSE 8080 8081

CMD ["torchserve", "--start", "--model-store", "/app/model_store", "--models", "email_reply=email_reply.mar", "--ncs", "--foreground"]
