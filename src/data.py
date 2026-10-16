"""Modulo per caricamento e preprocessing dei dati."""

import pandas as pd
import stylo_metrix as sm
from .config import MODELS


def load_data(path: str, models: list = None) -> pd.DataFrame:
    """Carica il CSV e filtra per modelli."""
    df = pd.read_csv(path)

    if models is None:
        models = MODELS

    # Filtra modelli
    df = df[df['modello'].isin(models)].copy()

    # Rimuovi errori
    df = df[~df['output'].str.startswith('ERRORE', na=False)]
    df = df.dropna(subset=['output'])
    df = df.reset_index(drop=True)

    print(f"Testi caricati: {len(df)}")
    print(df['modello'].value_counts())

    return df


def extract_stylometric_features(df: pd.DataFrame, lang: str = 'en') -> pd.DataFrame:
    """Estrae features stilometriche con StyloMetrix."""
    stylo = sm.StyloMetrix(lang)

    texts = df['output'].tolist()
    df_metrics = stylo.transform(texts)

    # Aggiungi colonne identificative
    df_metrics['modello'] = df['modello'].values
    if 'prompt_id' in df.columns:
        df_metrics['prompt_id'] = df['prompt_id'].values
    if 'topic' in df.columns:
        df_metrics['topic'] = df['topic'].values

    return df_metrics
