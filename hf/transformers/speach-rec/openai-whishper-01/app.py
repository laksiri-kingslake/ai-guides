import torch
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition",
               "openai/whisper-large-v3",
               torch_dtype=torch.float16,
               device="cuda:0")

pipe("https://cdn-media.huggingface.co/speech_samples/sample1.flac")

import gradio as gr

def transcribe(inputs):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please record an audio before submitting your request.")

    text = pipe(inputs, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
    return text

demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath"),
    ],
    outputs="text",
    title="Whisper Large V3: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the"
        " checkpoint [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) and 🤗 Transformers to transcribe audio files"
        " of arbitrary length."
    ),
    allow_flagging="never",
)

#demo.launch(share=True)
demo.launch(
    server_name="0.0.0.0",  # Allow all network interfaces
    server_port=7860,       # Explicitly set port
    share=False             # Disable temporary sharing service
)