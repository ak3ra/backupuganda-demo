import streamlit as st
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
from nltk.tokenize import sent_tokenize
from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio
from io import BytesIO

nltk.download("punkt")

st.set_page_config(page_title="Text Translation Demo", layout="wide")


@st.cache(allow_output_mutation=True)
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model, device


@st.cache(allow_output_mutation=True)
def load_tts_models():
    tacotron2 = Tacotron2.from_hparams(
        source="Sunbird/sunbird-lug-tts", savedir="tmpdir_tts"
    )
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder"
    )
    return tacotron2, hifi_gan


def translate_text(text, model, tokenizer, device, language_prefix):
    sentences = sent_tokenize(text)
    translated_text = ""

    for sentence in sentences:
        prefixed_sentence = f">>{language_prefix}<< {sentence}"
        inputs = tokenizer.encode(
            prefixed_sentence, return_tensors="pt", max_length=512, truncation=True
        ).to(device)
        translated_tokens = model.generate(
            inputs, max_length=512, num_return_sequences=1
        )
        translated_sentence = tokenizer.decode(
            translated_tokens[0], skip_special_tokens=True
        )
        translated_text += translated_sentence + " \n "

    return translated_text


model_name = "Sunbird/e2m_best_19_4_23"
tokenizer, model, device = load_model(model_name)
tacotron2, hifi_gan = load_tts_models()

st.title("Transcript Translation and Synthesis Demo")

language_prefix = st.selectbox("Select target language:", ["lug_hq"])

uploaded_file = st.file_uploader("Upload transcript as text file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    translated_text = translate_text(text, model, tokenizer, device, language_prefix)

    st.subheader("Original Transcript")
    st.write(text)

    st.subheader("Translated Text")
    translated_text_area = st.text_area("", translated_text, height=300)

    if st.button("Synthesize and Play Translated Transcript"):
        concatenated_waveforms = []

        for sentence in translated_text_area.split("\n"):
            # Generate the audio
            mel_output, mel_length, alignment = tacotron2.encode_text(sentence)
            waveforms = hifi_gan.decode_batch(mel_output)
            concatenated_waveforms.append(waveforms.squeeze(1))

        # Concatenate the waveforms
        concatenated_waveforms = torch.cat(concatenated_waveforms, dim=1)

        # Save the concatenated waveform to a temporary buffer
        buffer = BytesIO()
        torchaudio.save(buffer, concatenated_waveforms, 22050, format="wav")
        buffer.seek(0)

        # Display the audio player
        st.audio(buffer.read(), format="audio/wav", start_time=0)
