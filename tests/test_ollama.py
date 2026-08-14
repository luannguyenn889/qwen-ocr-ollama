# Nhiệm vụ: Kiểm tra kết nối cơ bản tới Ollama server và test tính năng sinh văn bản (text generation) đơn giản bằng model qwen3.5:4b.

if __name__ != "__main__":
    import unittest
    raise unittest.SkipTest("integration test; run directly with python tests/test_ollama.py")

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
