"""Download, pulizia e split del dataset German Credit.

Uso da terminale:
    python -m credit_xai.data            # scarica (se serve), pulisce, salva lo split
    python -m credit_xai.data --force    # riscarica il file grezzo
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_xai import config


def download_raw(path: Path | None = None, force: bool = False) -> Path:
    """Scarica il file grezzo da UCI se non è già presente."""
    path = path or config.RAW_FILE
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scarico {config.DATA_URL} ...")
    urllib.request.urlretrieve(config.DATA_URL, path)
    return path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Legge il file UCI (valori separati da spazi, senza intestazione)."""
    path = path or config.RAW_FILE
    df = pd.read_csv(path, sep=r"\s+", header=None, names=config.RAW_COLUMNS)
    if df.shape[1] != len(config.RAW_COLUMNS):
        raise ValueError(f"Attese {len(config.RAW_COLUMNS)} colonne, trovate {df.shape[1]}")
    return df


def age_group(age: pd.Series) -> pd.Series:
    """Fasce d'età usate per il fairness audit."""
    return pd.cut(
        age,
        bins=[0, 25, 35, 50, 120],
        labels=config.AGE_GROUPS,
    ).astype(str)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Decodifica i codici, separa sesso e stato civile, crea il target binario."""
    df = raw.copy()

    unknown = {
        col: sorted(set(df[col]) - set(mapping))
        for col, mapping in config.CODEBOOK.items()
        if set(df[col]) - set(mapping)
    }
    if unknown:
        raise ValueError(f"Codici non riconosciuti: {unknown}")

    for col, mapping in config.CODEBOOK.items():
        df[col] = df[col].map(mapping)

    status = df.pop("personal_status_sex").map(config.PERSONAL_STATUS_SEX)
    if status.isna().any():
        raise ValueError("Codici di personal_status_sex non riconosciuti")
    df["sex"] = status.str[0]
    df["personal_status"] = status.str[1]
    df["age_group"] = age_group(df["age"])

    # Nel file originale: 1 = buono, 2 = cattivo
    df[config.TARGET] = (df.pop("class") == 2).astype(int)

    for col in config.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col])
    return df


def split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split stratificato sul target, così train e test hanno la stessa quota di default."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[config.TARGET]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_dataset(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completa: download -> pulizia -> split -> salvataggio CSV."""
    download_raw(force=force_download)
    df = clean(load_raw())
    train, test = split(df)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DIR / "german_credit_clean.csv", index=False)
    train.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    print(
        f"Salvati {len(df)} record ({len(train)} train, {len(test)} test) "
        f"in {config.PROCESSED_DIR}. Tasso di default: {df[config.TARGET].mean():.1%}"
    )
    return train, test


def load_processed(name: str = "german_credit_clean") -> pd.DataFrame:
    """Carica un file già elaborato (german_credit_clean, train o test)."""
    path = config.PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} non trovato: esegui prima `python -m credit_xai.data`")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="riscarica il file grezzo")
    args = parser.parse_args()
    build_dataset(force_download=args.force)


if __name__ == "__main__":
    main()
