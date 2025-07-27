from flask import Flask, request, jsonify
from typing import List
import re
import fasttext  # всё работает как раньше
from huggingface_hub import hf_hub_download

app = Flask(__name__)

model_path = hf_hub_download("kenhktsui/code-natural-language-fasttext-classifier", "model.bin", cache_dir=".cached")
model = fasttext.load_model(model_path)

def replace_newlines(text: str) -> str:
    return re.sub(r"\n+", " ", text)

def predict(text_list: List[str]) -> List[dict]:
    text_list = [replace_newlines(text) for text in text_list]
    labels, scores = model.predict(text_list)
    return [
        {
            "label": label[0].replace("__label__", ""),
            "score": float(score[0])
        }
        for label, score in zip(labels, scores)
    ]

@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200

@app.route('/check', methods=['POST'])
def align():
    data = request.get_json(force=True)
    text = data.get("source")
    if not text or not isinstance(text, str):
        return jsonify({"error": "Expected JSON with a 'text' field as a string"}), 400

    result = predict([text])[0] 
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)