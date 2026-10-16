"""
Parti comuni a tutti gli script di generazione: modelli, ruoli per genere, parametri di decodifica,
chiamata all'API e scrittura incrementale del CSV.

Tutti i corpora (training, test, maschere, imitazione) devono passare da qui, così i parametri restano identici.
"""

import csv
import os
import re
import time
from pathlib import Path
from mistralai import Mistral
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# etichetta nel dataset -> (provider, id del modello)
MODELS = {
    "llama": ("groq", "llama-3.3-70b-versatile"),
    "mistral": ("mistral", "mistral-large-latest"),
    "openai": ("groq", "openai/gpt-oss-20b"),
    "qwen": ("groq", "qwen/qwen3-32b"),
}

# system prompt per genere
TOPIC_ROLES = {
    "narrative": "You are a professional creative writer specializing in short stories.",
    "argumentation": "You are an expert essayist specializing in formal and balanced argumentation.",
    "dialogue": "You are a professional scriptwriter specializing in natural and realistic dialogue.",
    "description": "You are a descriptive writer specializing in detailed environmental prose.",
}

# istruzione aggiunta al system prompt solo per Qwen (evita tag di ragionamento e altre lingue)
QWEN_SUFFIX = " Respond ONLY in English. Do not use reasoning tags or other languages."

GEN_PARAMS = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1500}

CSV_COLUMNS = ["modello", "prompt_id", "topic", "prompt", "output"]

_clients = {}


def _client(provider):
    """Crea il client del provider solo al primo uso (importare il modulo non richiede chiavi)."""
    if provider not in _clients:
        if provider == "groq":
            from groq import Groq
            _clients[provider] = Groq(api_key=os.environ["GROQ_API_KEY"])
        elif provider == "mistral":
            _clients[provider] = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        else:
            raise ValueError(f"Provider sconosciuto: {provider}")
    return _clients[provider]


def generate_text(model_key, topic, prompt, retries=3, system=None, qwen_suffix=True):
    """
    Genera un testo con il modello `model_key` per il genere `topic`.
    `system`: system prompt da usare al posto di quello del genere (TOPIC_ROLES).
    `qwen_suffix`: se False, non aggiunge a Qwen l'istruzione "Respond ONLY in English...".
    Ritenta con attesa crescente (limiti di frequenza); solleva l'ultima eccezione se fallisce.
    """
    provider, model_id = MODELS[model_key]
    system = (system or TOPIC_ROLES[topic]) + (QWEN_SUFFIX if model_key == "qwen" and qwen_suffix else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    for attempt in range(retries):
        try:
            if provider == "groq":
                resp = _client(provider).chat.completions.create(
                    model=model_id, messages=messages, stream=False, **GEN_PARAMS
                )
            else:
                resp = _client(provider).chat.complete(model=model_id, messages=messages, **GEN_PARAMS)
            text = resp.choices[0].message.content or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            if not text:
                raise ValueError("risposta vuota")
            return text
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def completed_keys(csv_path):
    """Insieme di (modello, genere, prompt_id) già presenti in un CSV, per poter riprendere dopo un'interruzione."""
    path = Path(csv_path)
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["modello"], r["topic"], int(r["prompt_id"])) for r in csv.DictReader(f) if r["output"]}


def append_row(csv_path, model_key, prompt_id, topic, prompt, output):
    path = Path(csv_path)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([model_key, prompt_id, topic, prompt, output])
