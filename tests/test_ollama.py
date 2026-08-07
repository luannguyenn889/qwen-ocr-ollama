from ollama import Client

client = Client(
    host="http://localhost:11434"
)

response = client.generate(
    model="qwen3.5:4b",
    prompt="Trả lời đúng một câu: Ollama đang hoạt động.",
    stream=False,
)

print(response.response)