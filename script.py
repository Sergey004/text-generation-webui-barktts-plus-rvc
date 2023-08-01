"""
Bark TTS extension for https://github.com/oobabooga/text-generation-webui/
All credit for the amazing tts model goes to https://github.com/suno-ai/bark/
All credit for the amazing RVC goes to https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/
"""
import hashlib
from http.client import IncompleteRead
import os
import time
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

# Read .env file
load_dotenv()

# Should change this environment variable before import bark
model_path = Path(os.environ.get('BARK_MODEL_PATH', 'extensions/bark_rvc_tts/models/'))
os.environ['XDG_CACHE_HOME'] = model_path.resolve().as_posix()

import nltk
import gradio as gr
import numpy as np
from bark import SAMPLE_RATE, preload_models, generate_audio
from bark.generation import ALLOWED_PROMPTS
from modules import shared
from extensions.bark_rvc_tts.rvc_infer import get_vc, vc_single
from scipy.io.wavfile import write as write_wav

nltk.download('punkt')

params = {
    'activate': True,
    'autoplay': False,
    'forced_speaker_enabled': False,
    'forced_speaker': 'Man',
    'show_text': False,
    'modifiers': [],
    'use_small_models': os.environ.get("USE_SMALL_MODELS", 'false').lower() == 'true',
    'use_cpu': os.environ.get("USE_CPU", 'false').lower() == 'true',
    'force_manual_download': False,
    'voice': 'v2/en_speaker_3',
    'sample_rate': SAMPLE_RATE,
    'text_temperature': 0.7,
    'waveform_temperature': 0.7,
}
sample_rate : SAMPLE_RATE

input_hijack = {
    'state': False,
    'value': ["", ""]
}

streaming_state = shared.args.no_stream
forced_modes = ["Man", "Woman", "Narrator"]
modifier_options = ["[laughter]", "[laughs]", "[sighs]", "[music]", "[gasps]", "[clears throat]"]
voice_presets = sorted(list(ALLOWED_PROMPTS))

# RVC Section
rvc_model_path = Path(os.environ.get('RVC_MODEL_PATH', 'extensions/bark_rvc_tts/Retrieval-based-Voice-Conversion-WebUI/weights/')) #Replace with your own
device="cuda:0"
is_half=True

index_rate = 0.75
f0up_key = 0
filter_radius = 3
rms_mix_rate = 0.25
protect = 0.33
resample_sr = sample_rate
f0method = os.environ.get("F0_METHOD", 'rmvpe').lower() #harvest or pm
rvc_index_path = Path(os.environ.get('RVC_INDEX_PATH', 'extensions/bark_rvc_tts/Retrieval-based-Voice-Conversion-WebUI/logs/')) #Replace with your own


def manual_model_preload():
    for model in ["text", "coarse", "fine", "text_2", "coarse_2", "fine_2"]:
        remote_url = f"https://dl.suno-models.io/bark/models/v0/{model}.pt"
        remote_md5 = hashlib.md5(remote_url.encode()).hexdigest()
        out_path = f"{os.path.expanduser('~/.cache/suno/bark_v0')}/{remote_md5}.pt"
        if not Path(out_path).exists():
            print(f"\t+ Downloading {model} model to {out_path}...")
            # we also have to do some user agent tomfoolery to get the download to work
            req = urllib.request.Request(remote_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0'})
            with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
                try:
                    data = response.read()
                except IncompleteRead as e:
                    data = e.partial
                out_file.write(data)
        else:
            print(f"\t+ {model} model already exists, skipping...")
    preload_models(
        text_use_gpu=not params['use_cpu'],
        text_use_small=params['use_small_models'],
        coarse_use_gpu=not params['use_cpu'],
        coarse_use_small=params['use_small_models'],
        fine_use_gpu=not params['use_cpu'],
        fine_use_small=params['use_small_models'],
        codec_use_gpu=not params['use_cpu']
    )


def toggle_activate(activate: bool) -> None:
    # Update params
    params.update({'activate': activate})

    # toggle streaming state
    if activate:
        shared.args.no_stream = True
        shared.processing_message = "*Is recording a voice message...*"
    else:
        shared.args.no_stream = streaming_state
        shared.processing_message = "*Is typing...*"


def generate_long_audio(ttstext: str) -> np.ndarray:
    """
    Split text into trunks to bypass 13 seconds limit

    - Args:
        ttstext (str): The text to convert to audio.

    - Returns:
        np.ndarray: The audio generated from the text.
    """
    # insert half second of silence in between trunks
    silence = np.zeros(int(0.5 * params['sample_rate']))

    # adapted from https://github.com/wsippel/bark_tts
    sentences = nltk.sent_tokenize(ttstext)
    chunks = ['']
    token_counter = 0
    for sentence in sentences:
        current_tokens = len(nltk.Text(sentence))
        if token_counter + current_tokens <= 250:
            token_counter = token_counter + current_tokens
            chunks[-1] = chunks[-1] + " " + sentence
        else:
            chunks.append(sentence)
            token_counter = current_tokens

    # Generate audio for each prompt
    audio_arrays = []
    for prompt in chunks:
        audio_array = generate_audio(
            text=prompt,
            text_temp=params['text_temperature'],
            waveform_temp=params['waveform_temperature'],
            history_prompt=params['voice'],
            )
        audio_arrays += [audio_array, silence.copy()]

    audio = np.array(np.concatenate(audio_arrays), dtype="float32")
    return audio


def input_modifier(string):

    # Remove autoplay from last reply
    if shared.is_chat() and len(shared.history['internal']) > 0:
        shared.history['visible'][-1][1] = shared.history['visible'][-1][1].replace('controls autoplay>', 'controls>')
    return string


def output_modifier(string):

    if not params['activate']:
        return string

    ttstext = string

    if params['modifiers']:
        ttstext = f"{' '.join(params['modifiers'])}: {ttstext}"

    if params['forced_speaker_enabled']:
        ttstext = f"{params['forced_speaker'].upper()}: {ttstext}"

    audio = generate_long_audio(ttstext)
    time_label = int(time.time())
    write_wav(f"extensions/bark_rvc_tts/generated_temp/{shared.character}_{time_label}.wav", params['sample_rate'], audio)
    output_file = Path(f"extensions/bark_rvc_tts/generated_temp/{shared.character}_{time_label}.wav")

    wav_opt = vc_single(0,output_file,f0up_key,None,f0method,rvc_index_path,index_rate, filter_radius=filter_radius, resample_sr=params['sample_rate'], rms_mix_rate=rms_mix_rate, protect=protect)
    write_wav(Path(f'extensions/bark_rvc_tts/generated/{shared.character}_{time_label}.wav'), resample_sr, wav_opt)

    autoplay = 'autoplay' if params['autoplay'] else ''
    if params['show_text']:
        string = f'<audio src="file/extensions/bark_tts/generated/{shared.character}_{time_label}.wav" controls {autoplay}></audio><br>{ttstext}'
    else:
        string = f'<audio src="file/extensions/bark_tts/generated/{shared.character}_{time_label}.wav" controls {autoplay}></audio>'

    return string


def setup():
    # tell the user what's going on
    print()
    print("== Loading Bark TTS extension ==")
    print("+ This may take a while on first run don't worry!")

    print("+ Creating directories (if they don't exist)...")
    if not Path("extensions/bark_rvc_tts/generated").exists():
        Path("extensions/bark_rvc_tts/generated").mkdir(parents=True)
    if not Path(model_path).exists():
        Path(model_path).mkdir(parents=True)
    print("+ Done!")

    # load models into extension directory so we don't clutter the pc
    print("+ Loading model...")
    if not params['force_manual_download']:
        try:
            preload_models(
                text_use_gpu=not params['use_cpu'],
                text_use_small=params['use_small_models'],
                coarse_use_gpu=not params['use_cpu'],
                coarse_use_small=params['use_small_models'],
                fine_use_gpu=not params['use_cpu'],
                fine_use_small=params['use_small_models'],
                codec_use_gpu=not params['use_cpu']
            )
        except ValueError as e:
            # for some reason the download fails sometimes, so we just do it manually
            # solution adapted from https://github.com/suno-ai/bark/issues/46
            print("\t+ Automatic download failed, trying manual download...")
            manual_model_preload()

    else:
        print("\t+ Forcing manual download...")
        manual_model_preload()
    get_vc(rvc_model_path, device, is_half)
    print("+ Done!")
    toggle_activate(params['activate'])
    print("== Bark TTS extension loaded ==\n\n")


def ui():
    with gr.Accordion("Bark TTS + RVC"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Autoplay')
            show_text = gr.Checkbox(value=params['show_text'], label='Show text')
            forced_speaker_enabled = gr.Checkbox(value=params['forced_speaker_enabled'], label='Forced speaker enabled')
        with gr.Row():
            forced_speaker = gr.Dropdown(forced_modes, label='Forced speaker', value=params['forced_speaker'])
            modifiers = gr.Dropdown(modifier_options, label='Modifiers', value=params['modifiers'], multiselect=True)
            voice = gr.Dropdown(voice_presets, label='Voice preset', value=params['voice'])
        with gr.Row():
            sample_rate = gr.Slider(minimum=18000, maximum=30000, value=params['sample_rate'], label='Sample rate')
            text_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=params['text_temperature'], label='Text temperature')
            audio_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=params['waveform_temperature'], label='Audio temperature')

    activate.change(toggle_activate, activate, None)
    autoplay.change(lambda x: params.update({'autoplay': x}), autoplay, None)
    show_text.change(lambda x: params.update({'show_text': x}), show_text, None)
    forced_speaker_enabled.change(lambda x: params.update({'forced_speaker_enabled': x}), forced_speaker_enabled, None)
    forced_speaker.change(lambda x: params.update({'forced_speaker': x}), forced_speaker, None)
    modifiers.change(lambda x: params.update({'modifiers': x}), modifiers, None)
    voice.change(lambda x: params.update({'voice': x}), voice, None)
    sample_rate.change(lambda x: params.update({'sample_rate': x}), sample_rate, None)
    text_temperature.change(lambda x: params.update({'text_temperature': x}), text_temperature, None)
    audio_temperature.change(lambda x: params.update({'waveform_temperature': x}), audio_temperature, None)
