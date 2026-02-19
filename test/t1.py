from leo import LeoLLMClient

llm = LeoLLMClient("openai/gpt-5-nano", "openrouter")

messages = [{"role": "user", "content": "hello, introduce yourself."}]

resp = llm.invoke(messages)

print(resp)
