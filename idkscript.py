import ollama

try:
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': 'Say hello, Judge!'},
    ])
    print(f"Response: {response['message']['content']}")
except Exception as e:
    print(f"Error: {e}")