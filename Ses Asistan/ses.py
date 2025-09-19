from ollama import chat

resp = chat(
    model="llama3.1",
    messages=[{"role":"user","content":"Arkadaşlarımla cips yiyeceğiz. Sence mısır cipsi mi almalıyız yoksa patates cipsi mi"}]
)
print(resp["message"]["content"])
