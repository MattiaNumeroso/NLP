"""
Genera i testi dell'esperimento di imitazione cross-model: un modello (imitatore) riceve 2 esempi "prototipici" di un altro
modello (bersaglio) e scrive un testo nuovo imitandone lo stile.

4 coppie imitatore->bersaglio x 4 generi x 8 prompt = 128 testi. I prompt sono i primi 8 di ogni genere di
dati/prompts_test.json (prompt_id 1-8 nell'output); gli esempi sono in dati/esempi_imitazione.json.

Nota: nello script originale (generate_cross_model_full.py) i prompt erano presi da test_set.csv, dove ogni prompt compare
4 volte (una per modello), quindi i testi di cross_model_test_set_full.csv usano solo 2 prompt distinti per genere
(ripetuti 4 volte). Qui gli 8 prompt sono tutti distinti.

Per sicurezza NON scrive su cross_model_test_set_full.csv: l'output di default è `corpus_imitazione_nuovo.csv`.

    python generate_imitation_corpus.py                           # solo riepilogo (dry run)
    python generate_imitation_corpus.py --run                     # genera i 128 testi
    python generate_imitation_corpus.py --run --pairs qwen->llama
"""

import argparse
import csv
import json
import time
from pathlib import Path

from common import generate_text

DATI = Path(__file__).resolve().parent / "dati"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "corpus_imitazione_nuovo.csv"

PAIRS = [("llama", "mistral"), ("mistral", "openai"), ("qwen", "llama"), ("openai", "qwen")]  # (imitatore, bersaglio)
N_PROMPTS = 8
SYSTEM_INSTRUCTION = (
    "You are an expert at stylistic mimicry. Your goal is to write a new text that PERFECTLY matches the rhythm, "
    "syntax, word choice, and overall writing style of the provided examples."
)
USER_TEMPLATE = """Study the following two examples of a specific writing style.

EXAMPLE 1:
{ex1}

EXAMPLE 2:
{ex2}

INSTRUCTION:
Write a NEW text following the prompt below. YOU MUST IMITATE THE EXACT STYLE OF THE EXAMPLES ABOVE. Do not mention that you are imitating, just write the text.

PROMPT:
{prompt}"""
COLONNE = ["imitation_pair", "imitator", "target", "topic", "prompt_id", "prompt", "output"]


def gia_generati(path):
    if not Path(path).exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["imitator"], r["target"], r["topic"], int(r["prompt_id"])) for r in csv.DictReader(f) if r["output"]}


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
    ap.add_argument("--pairs", nargs="+", default=None, help="coppie imitatore->bersaglio, es. qwen->llama")
    ap.add_argument("--topics", nargs="+", default=None)
    ap.add_argument("--pause", type=float, default=1.0)
    args = ap.parse_args()

    with open(DATI / "prompts_test.json", encoding="utf-8") as f:
        prompts = {t: v[:N_PROMPTS] for t, v in json.load(f).items()}
    with open(DATI / "esempi_imitazione.json", encoding="utf-8") as f:
        esempi = json.load(f)

    coppie = [p for p in PAIRS if args.pairs is None or f"{p[0]}->{p[1]}" in args.pairs]
    topics = args.topics or list(prompts)
    fatti = gia_generati(args.output)

    todo = []
    for imitatore, bersaglio in coppie:
        for topic in topics:
            for i, prompt in enumerate(prompts[topic]):
                if (imitatore, bersaglio, topic, i + 1) not in fatti:
                    todo.append((imitatore, bersaglio, topic, i + 1, prompt))

    print(f"Output: {args.output}")
    print(f"Coppie: {[f'{a}->{b}' for a, b in coppie]} | generi: {topics}")
    print(f"Già presenti: {len(fatti)} | da generare: {len(todo)}")
    if not args.run:
        print("Dry run: aggiungi --run per generare.")
        return

    ok = falliti = 0
    for n, (imitatore, bersaglio, topic, pid, prompt) in enumerate(todo, 1):
        ex1, ex2 = esempi[bersaglio][topic][:2]
        utente = USER_TEMPLATE.format(ex1=ex1, ex2=ex2, prompt=prompt)
        try:
            testo = generate_text(imitatore, topic, utente, retries=5, system=SYSTEM_INSTRUCTION, qwen_suffix=False)
        except Exception as e:
            falliti += 1
            print(f"[{n}/{len(todo)}] {imitatore}->{bersaglio} {topic} #{pid} FALLITO: {str(e)[:100]}", flush=True)
            continue
        aggiungi_riga(args.output, dict(imitation_pair=f"{imitatore}->{bersaglio}", imitator=imitatore, target=bersaglio,
                                        topic=topic, prompt_id=pid, prompt=prompt, output=testo))
        ok += 1
        print(f"[{n}/{len(todo)}] {imitatore}->{bersaglio} {topic} #{pid} ok ({len(testo)} caratteri)", flush=True)
        time.sleep(args.pause)

    print(f"\nCompletato: {ok} generati, {falliti} falliti. Rilancia per riprovare i falliti.")


if __name__ == "__main__":
    main()
