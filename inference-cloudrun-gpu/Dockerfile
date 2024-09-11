# gcloud beta run deploy --source . --port 11434 --no-cpu-throttling --cpu 8 --memory 32Gi --gpu 1 --gpu-type=nvidia-l4
FROM ollama/ollama
ENV HOME /root
WORKDIR /
RUN ollama serve & sleep 10 && ollama pull gemma2
ENTRYPOINT ["ollama","serve"]