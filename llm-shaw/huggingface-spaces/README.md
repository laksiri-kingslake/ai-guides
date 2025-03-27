# Guide to Set-up Huggingface Spaces

1. **Set-up Huggingface**
- Loging/Signup with  https://huggingface.co/
- Go to https://huggingface.co/spaces
- Click + New Space and follow the instructions

## File Structures

1. **README.md**
<br>The file structure has to be a below markdown text
```markdown
---
title: Vanilla Chat
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).
```

2. **app.py**
<br> Application logic implemented in here

```python
from transformers import pipeline, Conversation
import gradio as gr

chatbot = pipeline(model="facebook/blenderbot-400M-distill")

message_list = []
response_list = []

def vanilla_chatbot(message, history):
    conversation = Conversation(text=message, past_user_inputs=message_list, generated_responses=response_list)
    conversation = chatbot(conversation)

    return conversation.generated_responses[-1]

demo_chatbot = gr.ChatInterface(vanilla_chatbot, title="Vanilla Chatbot", description="Enter text to start chatting.")

demo_chatbot.launch()

```

3. **requirements.txt**
<br>Dependancy libriries
```text
gradio==3.39.0
transformers==4.31.0
torch==2.0.1 
```