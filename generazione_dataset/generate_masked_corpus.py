"""
Genera il test set mascherato (robustezza al tono): 4 maschere x 4 generi x 5 prompt x 4 modelli = 320 testi.

Ogni prompt è preceduto da un'istruzione di stile (la maschera): hemingway, academic, casual, gen_z.
I 5 prompt per genere sono prompt del corpus di training (vedi dati/prompts_maschere.json).

Come nel corpus originale (masked_test_set.csv), il system prompt è unico per tutti i generi:
"You are a professional creative writer."

Per sicurezza NON scrive su masked_test_set.csv: l'output di default è `corpus_maschere_nuovo.csv`.

    python generate_masked_corpus.py                    # solo riepilogo (dry run)
    python generate_masked_corpus.py --run              # genera i 320 testi
    python generate_masked_corpus.py --run --models openai --masks hemingway --topics narrative
"""

import argparse
import csv
import json
import time
from pathlib import Path

from common import MODELS, generate_text

DATI = Path(__file__).resolve().parent / "dati"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "corpus_maschere_nuovo.csv"
SYSTEM_GENERICO = "You are a professional creative writer."
COLONNE = ["modello", "topic", "mask", "prompt_id", "prompt", "output"]
MIN_CARATTERI = 50  # sotto questa lunghezza il testo è considerato fallito, come nello script originale


def load_config():
    with open(DATI / "prompts_maschere.json", encoding="utf-8") as f:
        return json.load(f)


def gia_generati(path):
    if not Path(path).exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["modello"], r["topic"], r["mask"], int(r["prompt_id"])) for r in csv.DictReader(f) if r["output"]}


def aggiungi_riga(path, riga):
    nuovo = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuovo:
            w.writerow(COLONNE)
        w.writerow([riga[c] for c in COLONNE])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="genera davvero (senza, mostra solo il riepilogo)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--topics", nargs="+", default=None)
    ap.add_argument("--masks", nargs="+", default=None)
    ap.add_argument("--pause", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config()
    maschere = cfg["maschere"]
    nomi = args.masks or list(maschere)
    topics = args.topics or list(cfg["prompts"])
    fatti = gia_generati(args.output)

    todo = []
    for mask in nomi:
        for topic in topics:
            for i, base in enumerate(cfg["prompts"][topic]):
                for model in args.models:
                    if (model, topic, mask, i + 1) not in fatti:
                        todo.append((model, topic, mask, i + 1, maschere[mask] + base))

    print(f"Output: {args.output}")
    print(f"Maschere: {nomi} | generi: {topics} | modelli: {args.models}")
    print(f"System prompt: {SYSTEM_GENERICO!r}")
    print(f"Già presenti: {len(fatti)} | da generare: {len(todo)}")
    if not args.run:
        print("Dry run: aggiungi --run per generare.")
        return

    ok = falliti = 0
    for n, (model, topic, mask, pid, prompt) in enumerate(todo, 1):
        try:
            testo = generate_text(model, topic, prompt, system=SYSTEM_GENERICO)
            if len(testo) <= MIN_CARATTERI:
                raise ValueError(f"testo troppo corto ({len(testo)} caratteri)")
        except Exception as e:
            falliti += 1
            print(f"[{n}/{len(todo)}] {mask} {topic} {model} #{pid} FALLITO: {str(e)[:100]}", flush=True)
            continue
        aggiungi_riga(args.output, dict(modello=model, topic=topic, mask=mask, prompt_id=pid, prompt=prompt, output=testo))
        ok += 1
        print(f"[{n}/{len(todo)}] {mask} {topic} {model} #{pid} ok ({len(testo)} caratteri)", flush=True)
        time.sleep(args.pause)

    print(f"\nCompletato: {ok} generati, {falliti} falliti. Rilancia per riprovare i falliti.")


if __name__ == "__main__":
    main()
