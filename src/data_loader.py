import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    try:
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin1')

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        return df

    except Exception as e:
        raise RuntimeError(f"Erro ao carregar dados: {e}")