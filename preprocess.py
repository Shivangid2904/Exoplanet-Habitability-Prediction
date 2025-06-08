import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, header=88)  # Header starts from row 89

    required_columns = [
        "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "pl_insol",
        "pl_orbeccen", "st_teff", "st_rad", "st_mass", "st_met", "sy_dist"
    ]
    df = df.dropna(subset=required_columns)

    df['is_habitable'] = (
        (df['pl_rade'] >= 0.5) & (df['pl_rade'] <= 2.5) &
        (df['pl_eqt'] >= 180) & (df['pl_eqt'] <= 310) &
        (df['pl_insol'] >= 0.3) & (df['pl_insol'] <= 1.8)
    ).astype(int)

    return df
