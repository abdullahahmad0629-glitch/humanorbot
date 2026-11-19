"""
predict.py - classify a text with the base model AND the fine-tuned model.

The model is loaded once. The base model's answer is obtained by switching
the LoRA adapter off, so we don't need two copies of the 1.5B model.

Usage:
    python src/predict.py "Some text to check"
    python src/predict.py                      # type texts one by one
    python src/predict.py --models-dir /kaggle/working/outputs "text"

Output (JSON, easy to send to the React front end later):
    {"text": "...", "predictions": {
        "base":      {"label": "human", "ai_probability": 0.41},
        "finetuned": {"label": "ai",    "ai_probability": 0.97}}}
"""

import argparse
import json
import os

from peft import PeftModel

from preprocess import clean_text
from train import get_device, load_model, predict_ai_prob


class Predictor:
    def __init__(self, models_dir="outputs"):
        with open(os.path.join(models_dir, "results.json")) as f:
            settings = json.load(f)["settings"]
        self.max_len = settings["max_len"]
        self.device = get_device()

        self.tokenizer, base = load_model(settings["model_name"], self.device)
        self.model = PeftModel.from_pretrained(base, os.path.join(models_dir, "lora_adapter"))
        self.model.eval()

    def _result(self, prob):
        return {"label": "ai" if prob >= 0.5 else "human", "ai_probability": round(prob, 4)}

    def predict(self, text):
        texts = [clean_text(text)]

        with self.model.disable_adapter():          # base model (LoRA off)
            base_prob = predict_ai_prob(self.model, self.tokenizer, texts,
                                        self.max_len, 1, self.device)[0]
        ft_prob = predict_ai_prob(self.model, self.tokenizer, texts,   # fine-tuned (LoRA on)
                                  self.max_len, 1, self.device)[0]

        return {"text": text,
                "predictions": {"base": self._result(float(base_prob)),
                                "finetuned": self._result(float(ft_prob))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default=None)
    parser.add_argument("--models-dir", default="outputs")
    args = parser.parse_args()

    predictor = Predictor(args.models_dir)

    if args.text:
        print(json.dumps(predictor.predict(args.text), indent=2))
    else:
        print("Type a text and press Enter (empty line to quit).")
        while True:
            text = input("> ").strip()
            if not text:
                break
            print(json.dumps(predictor.predict(text), indent=2))
