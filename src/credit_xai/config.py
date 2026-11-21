"""Percorsi, costanti e schema del dataset German Credit (Statlog, UCI)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

MODEL_FILE = MODELS_DIR / "model.joblib"
METRICS_FILE = REPORTS_DIR / "metrics.json"

RAW_FILE = RAW_DIR / "german.data"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "default"  # 1 = cattivo pagatore, 0 = buon pagatore

# Nomi delle 20 colonne originali + target, nell'ordine del file UCI.
RAW_COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "class",
]

# Decodifica dei codici categorici (documentazione UCI, file german.doc).
CODEBOOK: dict[str, dict[str, str]] = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0-200 DM",
        "A13": ">= 200 DM",
        "A14": "nessun conto",
    },
    "credit_history": {
        "A30": "nessun credito / tutti rimborsati",
        "A31": "rimborsati in questa banca",
        "A32": "crediti in corso regolari",
        "A33": "ritardi in passato",
        "A34": "conto critico / altri crediti",
    },
    "purpose": {
        "A40": "auto nuova",
        "A41": "auto usata",
        "A42": "mobili",
        "A43": "radio/TV",
        "A44": "elettrodomestici",
        "A45": "riparazioni",
        "A46": "istruzione",
        "A47": "vacanza",
        "A48": "riqualificazione",
        "A49": "business",
        "A410": "altro",
    },
    "savings": {
        "A61": "< 100 DM",
        "A62": "100-500 DM",
        "A63": "500-1000 DM",
        "A64": ">= 1000 DM",
        "A65": "sconosciuto / nessuno",
    },
    "employment_since": {
        "A71": "disoccupato",
        "A72": "< 1 anno",
        "A73": "1-4 anni",
        "A74": "4-7 anni",
        "A75": ">= 7 anni",
    },
    "other_debtors": {
        "A101": "nessuno",
        "A102": "co-richiedente",
        "A103": "garante",
    },
    "property": {
        "A121": "immobile",
        "A122": "assicurazione vita / risparmio edilizio",
        "A123": "auto o altro",
        "A124": "nessuna / sconosciuta",
    },
    "other_installment_plans": {"A141": "banca", "A142": "negozi", "A143": "nessuno"},
    "housing": {"A151": "affitto", "A152": "proprietà", "A153": "gratuito"},
    "job": {
        "A171": "disoccupato / non qualificato non residente",
        "A172": "non qualificato residente",
        "A173": "qualificato / impiegato",
        "A174": "dirigente / autonomo / altamente qualificato",
    },
    "telephone": {"A191": "no", "A192": "sì"},
    "foreign_worker": {"A201": "sì", "A202": "no"},
}

# La colonna originale mescola sesso e stato civile: la separiamo.
PERSONAL_STATUS_SEX = {
    "A91": ("M", "divorziato/separato"),
    "A92": ("F", "divorziata/separata/sposata"),
    "A93": ("M", "single"),
    "A94": ("M", "sposato/vedovo"),
    "A95": ("F", "single"),
}

NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "people_liable",
]

# Attributi protetti: NON li usiamo come feature del modello, ma li teniamo
# per il fairness audit. L'età invece resta tra le feature (fattore di rischio
# considerato legittimo nel credito): la scelta è motivata nella model card
# e controllata nel fairness audit tramite age_group.
SENSITIVE_ATTRIBUTES = ["sex", "age_group", "foreign_worker"]
AGE_GROUPS = ["<=25", "26-35", "36-50", ">50"]

# Esclusi dal modello: attributi protetti + personal_status (proxy del sesso).
EXCLUDED_FROM_MODEL = ["sex", "personal_status", "foreign_worker", "age_group"]

# Feature su cui il cliente può agire: le sole che i controfattuali possono cambiare.
# Escluse quelle immutabili (età) o non realistiche da cambiare (storia creditizia,
# lavoro, anni di residenza, persone a carico).
ACTIONABLE_FEATURES = [
    "duration_months",
    "credit_amount",
    "checking_status",
    "savings",
    "installment_rate",
    "other_debtors",
    "existing_credits",
    "other_installment_plans",
    "purpose",
]

# Modifiche non plausibili come consiglio al cliente, scartate dai controfattuali.
# Es.: nel German Credit "nessun conto" ha il rischio più basso, ma "chiuda il conto
# corrente" non è un consiglio sensato; aprire altri crediti nemmeno.
IMPLAUSIBLE_TARGETS = {"checking_status": ["nessun conto"]}
ONLY_DECREASE = ["existing_credits", "installment_rate"]

# Matrice dei costi della documentazione UCI: concedere credito a un cattivo
# pagatore (falso negativo) costa 5, rifiutarlo a un buon pagatore costa 1.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0

# Nomi leggibili delle feature, usati nelle spiegazioni.
FEATURE_LABELS = {
    "checking_status": "saldo del conto corrente",
    "duration_months": "durata del prestito",
    "credit_history": "storia creditizia",
    "purpose": "scopo del prestito",
    "credit_amount": "importo richiesto",
    "savings": "risparmi",
    "employment_since": "anzianità lavorativa",
    "installment_rate": "rata in % del reddito",
    "other_debtors": "altri debitori o garanti",
    "residence_since": "anni nella residenza attuale",
    "property": "proprietà",
    "age": "età",
    "other_installment_plans": "altri finanziamenti in corso",
    "housing": "situazione abitativa",
    "existing_credits": "crediti già aperti",
    "job": "tipo di lavoro",
    "people_liable": "persone a carico",
    "telephone": "telefono registrato",
}
