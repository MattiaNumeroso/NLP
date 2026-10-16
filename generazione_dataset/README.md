# Generazione del dataset

Versione ordinata degli script che hanno generato il corpus.

| File | Cosa fa |
|---|---|
| `common.py` | modelli, system prompt per genere, parametri (temperature 0.7, top-p 0.9, 1500 token), chiamata API con retry, scrittura CSV |
| `generazione.py` | motore comune: legge un JSON di prompt, genera i testi mancanti, riprende se interrotto, non salva i fallimenti |
| `generate_training_corpus.py` | corpus di training: 50 prompt x 4 generi x 4 modelli = 800 testi |
| `generate_test_corpus.py` | test set: 10 prompt nuovi x 4 generi x 4 modelli = 160 testi |
| `generate_masked_corpus.py` | test set mascherato: 4 maschere x 4 generi x 5 prompt x 4 modelli = 320 testi |
| `generate_imitation_corpus.py` | imitazione cross-model: 4 coppie imitatore->bersaglio x 4 generi x 8 prompt = 128 testi (2 esempi prototipici del bersaglio nel prompt) |
| `dati/prompts_training.json` | 200 prompt di training `{genere: [prompt_0, ...]}` |
| `dati/prompts_maschere.json` | le 4 maschere (prefissi di stile) e i 5 prompt base per genere (prompt del training) |
| `dati/esempi_imitazione.json` | i 2 testi prototipici per modello e genere usati come esempi nel prompt di imitazione |
| `dati/prompts_test.json` | 40 prompt di test (10 per genere), identici a `test_set.csv` |

```bash
cd generazione_dataset
python generate_training_corpus.py            # dry run
python generate_training_corpus.py --run      # scrive corpus_training_nuovo.csv (non tocca corpus_training.csv)
python generate_masked_corpus.py --run          # scrive corpus_maschere_nuovo.csv, 320 testi (non tocca masked_test_set.csv)
python generate_imitation_corpus.py --run       # scrive corpus_imitazione_nuovo.csv, 128 testi (non tocca cross_model_test_set_full.csv)
python generate_test_corpus.py --run          # scrive corpus_test_nuovo.csv, 160 testi (non tocca test_set.csv)
```

Chiavi API: `GROQ_API_KEY` e `MISTRAL_API_KEY` nel file `.env` nella radice del repository.

## Storia del corpus originale (`corpus_training.csv`, prima `narrative.csv`)

Il corpus di training fu generato in tre passi con gli stessi modelli, parametri e system prompt per genere:
prompt 0-5 (`benchmark_complete.py`), prompt 6-24 (`generate_new_prompts.py`), prompt 25-49 (`generate_800.py`).
Gli script originali non sono inclusi nel repository.
Stato attuale: Llama 3.3 70B e Qwen3 32B non sono più disponibili su Groq e Mistral Large dà 403 con la chiave attuale;
`openai/gpt-oss-20b` funziona.

## Nota sull'imitazione
Gli script originali (`generate_cross_model_full.py`) prendevano i prompt da `test_set.csv`, dove ogni prompt compare una volta
per modello: `cross_model_test_set_full.csv` contiene quindi solo 2 prompt distinti per genere e coppia (ripetuti 4 volte),
non 8. `generate_imitation_corpus.py` usa 8 prompt distinti.
