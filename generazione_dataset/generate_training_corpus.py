"""
Genera il corpus di training: 50 prompt x 4 generi x 4 modelli = 800 testi.

Per sicurezza NON scrive su corpus_training.csv: il file di output è `corpus_training_nuovo.csv` (cambia con --output).

    python generate_training_corpus.py                         # solo riepilogo (dry run)
    python generate_training_corpus.py --run                   # genera tutto
    python generate_training_corpus.py --run --models openai   # un solo modello
    python generate_training_corpus.py --run --topics dialogue --prompts 0-9
"""

from pathlib import Path

from generazione import DATI, run_cli

if __name__ == "__main__":
    run_cli(__doc__, DATI / "prompts_training.json", Path(__file__).resolve().parent / "corpus_training_nuovo.csv")
