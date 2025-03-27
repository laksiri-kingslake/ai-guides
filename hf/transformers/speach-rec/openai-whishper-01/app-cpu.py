from transformers import pipeline  # torch is still needed but will use CPU

# Remove GPU-specific parameters and dtype specification
pipe = pipeline("automatic-speech-recognition",
               "openai/whisper-large-v3")  # Device defaults to CPU

# Optional test (will be slower on CPU)
# pipe("https://cdn-media.huggingface.co/speech_samples/sample1.flac")

import gradio as gr

def transcribe(inputs):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please record an audio before submitting your request.")

    # Remove return_timestamps if you want faster transcription
    text = pipe(inputs, generate_kwargs={"task": "transcribe"})["text"]
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

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)