import os
import joblib

from dotenv import load_dotenv
from src.data_loader import load_data
from src.feature_engineering import transform_data
from src.deploy import save_to_rds

import umap


# ==============================
# CONFIG
# ==============================

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATA_PATH = os.getenv("S3_PATH")

ENDPOINT = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# ==============================
# MAIN
# ==============================
def main():

    print("🔹 [1/8] Carregando dados...")
    df = load_data(DATA_PATH)

    print("🔹 [2/8] Feature engineering...")
    df = transform_data(df)

    features = [
        'revenue_velocity',
        'total_items',
        'items_velocity',
        'return_orders',
        'product_loyalty'
    ]

    X = df[features].copy()

    print("🔹 [3/8] Carregando modelos...")

    rf = joblib.load("models/rf.pkl")
    ohe = joblib.load("models/ohe.pkl")
    kmeans = joblib.load("models/kmeans.pkl")

    print("🔹 [4/8] Gerando embedding...")

    leafs = rf.apply(X)
    embedding = ohe.transform(leafs)

    print("🔹 [5/8] Recalculando UMAP...")

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        random_state=42
    )

    embedding_reduced = reducer.fit_transform(embedding)

    print("🔹 [6/8] Gerando clusters...")

    df['cluster'] = kmeans.predict(embedding_reduced)

    print("🔹 [7/8] Salvando no RDS...")

    save_to_rds(df, ENDPOINT)

    print("✅ Predição finalizada!")
    print("✅ Dados salvos no RDS")


if __name__ == "__main__":
    main()