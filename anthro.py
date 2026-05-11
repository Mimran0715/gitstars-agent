import anthropic

client = anthropic.Anthropic(
    base_url='',
    api_key='ollama',  # required but ignored, 
)
message = client.messages.create(
    model='gpt-oss:20b',
    max_tokens=1024,
    messages=[
        {'role': 'user', 'content': 'Write a function to check if a number is prime', 
        }
    ],
)
print(message.content[0].text)
