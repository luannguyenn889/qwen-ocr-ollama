# pyrefly: ignore [missing-import]
from ollama import Client

print("Đang gọi Ollama...", flush=True)

client = Client(host="http://localhost:11434")

response = client.generate(
    model="qwen3.5:4b",
    prompt="Trả lời đúng một câu: Ollama đang hoạt động.",
    think=False,
    stream=False,
    options={
        "num_predict": 32,
        "temperature": 0,
    },
)

print(response.response)