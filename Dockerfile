FROM ollama/ollama:latest

# Khởi động Ollama tạm thời để thực hiện tải (pull) model ngay trong quá trình build image
RUN ollama serve & \
    sleep 5 && \
    ollama pull qwen3.5:4b

# Khi chạy container, nó sẽ tự động chạy Ollama với model đã có sẵn bên trong
ENTRYPOINT ["ollama", "serve"]
