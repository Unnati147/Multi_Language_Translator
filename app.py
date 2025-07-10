import torch
import gradio as gr
import json
from transformers import pipeline
from gtts import gTTS
import tempfile

# Load translation pipeline
text_translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", torch_dtype=torch.bfloat16)

# Load language mapping
with open('language.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get FLORES-200 code
def get_FLORES_code_from_language(language):
    for entry in data['languages']:
        if entry['language'].lower() == language.lower():
            return entry['code']
    return None

# Translate from TEXT input
def translate_from_text(text_input, destination_language):
    dest_code = get_FLORES_code_from_language(destination_language)
    if not dest_code:
        return "Language not supported", None
    translated = text_translator(text_input, src_lang="eng_Latn", tgt_lang=dest_code)[0]['translation_text']

    # Convert to speech
    tts = gTTS(translated)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)

    return translated, temp_audio.name

# List of languages
dropdown_languages = sorted([entry["language"] for entry in data["languages"]])

gr.close_all()

# Interface
with gr.Blocks() as demo:
    # Custom title and subtitle styling
    gr.Markdown("""
        <h1 style='text-align: center; color: #1e40af;'>VoiceGlobe</h1>
        <h3 style='text-align: center; color: #2563eb;'> Multi Language Translator</h3>
        <p style='text-align: center; color: #374151; font-size: 16px;'><b>Type in English and get translated text with audio in your selected language.</b></p>
    """)

    with gr.Row():
        text_input = gr.Textbox(lines=5, label="Type text in English", placeholder="Enter your sentence here...")
        language_input = gr.Dropdown(dropdown_languages, label="Select Destination Language")

    translated_text = gr.Textbox(label="Translated Text")
    translated_audio = gr.Audio(label="🔊 Listen Translation")

    # Custom styled button
    translate_button = gr.Button(" Translate", elem_id="translate-btn")

    def process_input(text_input, language_input):
        return translate_from_text(text_input, language_input)

    translate_button.click(
        fn=process_input,
        inputs=[text_input, language_input],
        outputs=[translated_text, translated_audio]
    )

    # Inject custom CSS for button styling
    gr.HTML("""
        <style>
            #translate-btn {
                background-color: #2563eb;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                transition: 0.3s ease-in-out;
            }
            #translate-btn:hover {
                background-color: #1e40af;
                transform: scale(1.05);
                cursor: pointer;
            }
        </style>
    """)

demo.launch()
