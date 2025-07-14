from llama_cpp import Llama

llm = Llama(
    model_path="/Users/anupam.singhdeo/Softwares/llm/models/mistral/mistral.q4.gguf",
    n_ctx=2048,
    n_threads=6  # Tune based on your Mac's performance
)

response = llm(
    "Q: What is Kubernetes?\nA:",
    max_tokens=100,
    stop=["Q:"]
)

print(response["choices"][0]["text"].strip())
