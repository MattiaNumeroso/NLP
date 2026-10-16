# Configurazione del progetto

# Gruppi di features per analisi separate
FEATURE_GROUPS = {
    'POS': 'POS_',           # Part-of-speech ratios
    'Lessico': 'ST_',        # Statistiche lessicali (TTR, lunghezza frasi, etc.)
    'Sintassi': 'SY_',       # Pattern sintattici
    'Punteggiatura': 'PUNCT', # Frequenza punteggiatura
    'Discorso': 'PS_',       # Marcatori del discorso
    'Tempi_verbali': 'VT_',  # Tempi verbali
    'Grammatica': 'G_',      # Voce attiva/passiva, etc.
    'Pronomi': 'L_',         # Pronomi e altre features lessicali
}

# Combinazioni predefinite per esperimenti
EXPERIMENTS = {
    'all': list(FEATURE_GROUPS.keys()),
    'core': ['POS', 'Lessico', 'Sintassi'],
    'style': ['POS', 'Punteggiatura', 'Tempi_verbali'],
    'discourse': ['Discorso', 'Sintassi', 'Grammatica'],
}

# Modelli da analizzare
MODELS = ['llama', 'mistral', 'openai', 'qwen']

# Colori per visualizzazione
MODEL_COLORS = {
    'llama': '#1f77b4',    # blu
    'openai': '#d62728',   # rosso
    'mistral': '#ff7f0e',  # arancione
    'qwen': '#000000',     # nero
}
