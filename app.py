from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

app = Flask(__name__)


def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model, device


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
        translated_text += translated_sentence + " "

    return translated_text


model_name = "Sunbird/sunbird-en-mul"
tokenizer, model, device = load_model(model_name)


@app.route("/translate", methods=["POST"])
def translate():
    input_data = request.get_json()
    text = input_data["text"]
    language_prefix = input_data["language_prefix"]
    translated_text = translate_text(text, model, tokenizer, device, language_prefix)
    return jsonify(translated_text=translated_text)


if __name__ == "__main__":
    app.run(debug=True)
