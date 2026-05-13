# src/deploy.py
from sqlalchemy import create_engine
import pandas as pd


def save_to_rds(df: pd.DataFrame, endpoint: str):

    engine = create_engine(endpoint)

    df.to_sql(
        'rankers',
        con=engine,
        if_exists='replace',
        index=False
    )

    engine.dispose()