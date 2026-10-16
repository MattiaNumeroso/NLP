"""
Motore comune degli script di generazione: legge i prompt da un JSON {genere: [prompt, ...]}, genera i testi mancanti
con i 4 modelli e li aggiunge a un CSV. Riprende da dove si era fermato e non salva le chiamate fallite.

Gli script `generate_*_corpus.py` lo usano passando il file dei prompt, l'output di default e l'id del primo prompt.
"""

import argparse
import json
import time
from pathlib import Path

from common import MODELS, append_row, completed_keys, generate_text

DATI = Path(__file__).resolve().parent / "dati"


def load_prompts(path):
    """Legge {genere: [prompt_0, prompt_1, ...]} dal JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_range(text, default):
    """'0-9' -> range(0, 10); '5' -> range(5, 6); None -> default."""
    if text is None:
        return default
    a, _, b = text.partition("-")
    return range(int(a), int(b or a) + 1)


def run_cli(descrizione, prompts_path, default_output, first_id=0):
    ap = argparse.ArgumentParser(description=descrizione, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="genera davvero (senza, mostra solo il riepilogo)")
    ap.add_argument("--output", default=str(default_output))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--topics", nargs="+", default=None)
    ap.add_argument("--prompts", default=None, help=f"intervallo di prompt_id, es. {first_id}-{first_id + 4}")
    ap.add_argument("--pause", type=float, default=0.5, help="secondi di pausa tra le chiamate")
    args = ap.parse_args()

    prompts = load_prompts(prompts_path)
    topics = args.topics or list(prompts)
    done = completed_keys(args.output)

    todo = []
    for t in topics:
        ids = parse_range(args.prompts, range(first_id, first_id + len(prompts[t])))
        for pid in ids:
            if not first_id <= pid < first_id + len(prompts[t]):
                continue
            for m in args.models:
                if (m, t, pid) not in done:
                    todo.append((m, t, pid, prompts[t][pid - first_id]))

    print(f"Prompt: {Path(prompts_path).name} | output: {args.output}")
    print(f"Modelli: {args.models} | generi: {topics}")
    print(f"Già presenti: {len(done)} | da generare: {len(todo)}")
    if not args.run:
        print("Dry run: aggiungi --run per generare.")
        return

    ok = failed = 0
    for n, (model, topic, pid, prompt) in enumerate(todo, 1):
        try:
            text = generate_text(model, topic, prompt)
        except Exception as e:
            failed += 1
            print(f"[{n}/{len(todo)}] {model} {topic} #{pid} FALLITO: {str(e)[:120]}", flush=True)
            continue
        append_row(args.output, model, pid, topic, prompt, text)
        ok += 1
        print(f"[{n}/{len(todo)}] {model} {topic} #{pid} ok ({len(text)} caratteri)", flush=True)
        time.sleep(args.pause)

    print(f"\nCompletato: {ok} generati, {failed} falliti. Rilancia per riprovare i falliti.")
