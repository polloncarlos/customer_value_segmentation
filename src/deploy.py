from sqlalchemy import create_engine
import pandas as pd


def save_to_rds(df: pd.DataFrame, endpoint: str):

    print("🔹 Criando engine...")
    engine = create_engine(endpoint)

    print("🔹 Iniciando upload...")
    
    df.to_sql(
        'rankers',
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=1000,
        method='multi'
    )

    print("🔹 Finalizando conexão...")
    engine.dispose()

    print("✅ Upload concluído!")