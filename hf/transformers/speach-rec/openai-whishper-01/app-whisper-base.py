from transformers import pipeline
import gradio as gr

# Initialize the pipeline with whisper-base
pipe = pipeline(
    "automatic-speech-recognition",
    "openai/whisper-base",
    # Optional: Add chunk_length_s=30 for long audio processing
    chunk_length_s=30,  # Process in 30s chunks
    stride_length_s=[5, 3]  # Overlap chunks for better continuity
)

pipe.model.config.forced_decoder_ids = None

def transcribe(inputs):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please record audio first.")
    
    # Process audio (remove return_timestamps for faster performance)
    result = pipe(inputs, generate_kwargs={"task": "transcribe"})
    return result["text"]

demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs="text",
    title="Whisper Base: Speech to Text",
    description=(
        "Real-time transcription using OpenAI's Whisper Base model. "
        "Works best with clear English speech under 1 minute."
    ),
    allow_flagging="never",
)



demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)