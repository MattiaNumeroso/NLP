"""
Genera il test set, con prompt mai visti in training: 10 prompt x 4 generi x 4 modelli = 160 testi.

Per sicurezza NON scrive su test_set.csv: l'output di default è `corpus_test_nuovo.csv`.

    python generate_test_corpus.py                       # solo riepilogo (dry run)
    python generate_test_corpus.py --run                 # genera i 160 testi
    python generate_test_corpus.py --run --models openai --topics narrative
"""

from pathlib import Path

from generazione import DATI, run_cli

if __name__ == "__main__":
    run_cli(__doc__, DATI / "prompts_test.json", Path(__file__).resolve().parent / "corpus_test_nuovo.csv")
